# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///

"""
Simulates various backoff strategies for contending writes over the network.

The setup:
- many clients, starting at the same time, request over the network to write a particular database row
- if writes contend, only one commits
- various concurrency controls are modeled

You want to keep low
- the time until all writes commit
- the total number of requests

You can keep the completion time low by making clients retry rapidly.
But then writes often contend, so the request count is high.
You can keep the request count low by making clients retry sporadically.
But then the server is often idle, so the completion time is high.
So there’s a tradeoff.

There's a famous aws blog post and simulation script about this:
- https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter
- https://github.com/aws-samples/aws-arch-backoff-simulator

This script is my own exploration.
"""

import heapq
import random
from dataclasses import dataclass, field
from itertools import count, repeat
from typing import Callable, Iterator

from enum import StrEnum


@dataclass
class MaybeCommitPayload:
    version: int
    client_id: int


@dataclass(frozen=True)
class Message:
    """
    Message from simulated resource to simulation loop.

    Read as: "please schedule this: after delay, call target with payload,
    and tell it to reply by calling reply_target."
    """

    delay: float
    target: Callable[..., TargetResult]
    payload: int | MaybeCommitPayload | None = None
    reply_target: Callable | None = None


class EventType(StrEnum):
    # These happen in all concurrency controls.
    # So use them for a control-independent analysis.
    CLIENT_REQUESTS_WRITE = "client_requests_write"
    CLIENT_BACKS_OFF = "client_backs_off"
    SERVER_COMMITS = "server_commits"
    # These happen in some controls but not others.
    CLIENT_REQUESTS_VERSION = "client_requests_version"
    SERVER_ACCEPTS = "server_accepts"
    SERVER_REJECTS = "server_rejects"
    SERVER_REPORTS_VERSION = "server_reports_version"
    SERVER_TENTATIVELY_WRITES = "server_tentatively_writes"
    SERVER_ABORTS = "server_aborts"


@dataclass
class Event:
    """
    A simulation produces a timed sequence of events: everything of interest that happened, for later analysis.
    """

    event_type: EventType
    client_id: int | None = None
    event_detail: str | None = None  # human-readable, free-form


type TargetResult = tuple[Event, Message | None]


@dataclass(order=True, frozen=True)
class Todo:
    """
    Scheduled unit of work for the simulation loop.

    Read as: "at time, call target with payload and tell it to reply by calling reply_target."
    """

    time: float
    target: Callable[..., TargetResult] = field(compare=False)
    payload: int | None = field(compare=False, default=None)
    reply_target: Callable | None = field(compare=False, default=None)


class Network:
    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma

    def delay(self) -> float:
        return max(0, random.gauss(self.mu, self.sigma))


class ReadWriteOCCServer:
    """
    Simulates a server receiving contending write requests, using read-then-write optimistic concurrency control.

    The server stores a version number.
    A client first reads the version, then requests a write, passing the version.
    The server tentatively writes.
    Then it checks the version again:
        if different, it aborts
        if the same, it commits and increments the version

    The writes are variable-duration (not essential, but more realistic).
    """

    def __init__(self, network: Network, write_mu: float, write_sigma: float):
        self.version = 0
        self.network = network
        self.write_mu = write_mu
        self.write_sigma = write_sigma

    def write_duration(self) -> float:
        return max(0, random.gauss(self.write_mu, self.write_sigma))  # TODO: maybe abs instead of max, else 0 gets too much mass?

    def handle_read(self, client_id: int, read_response_handler: Callable) -> TargetResult:
        return (
            Event(
                EventType.SERVER_REPORTS_VERSION,
                client_id,
                f"version={self.version}",
            ),
            Message(
                delay=self.network.delay(),
                target=read_response_handler,
                payload=self.version,
            ),
        )

    def handle_write(self, payload: MaybeCommitPayload, abort_handler: Callable) -> TargetResult:
        return (
            Event(
                EventType.SERVER_TENTATIVELY_WRITES,
                payload.client_id,
            ),
            Message(
                delay=self.write_duration(),  # server-internal, so no network delay
                target=self.maybe_commit,
                payload=payload,
                reply_target=abort_handler,
            ),
        )

    def maybe_commit(self, payload: MaybeCommitPayload, abort_handler: Callable) -> TargetResult:
        assert self.version >= payload.version, f"version has decreased: {self.version=} < {payload.version=}"

        if self.version > payload.version:
            # another write committed in the meantime
            return (
                Event(
                    EventType.SERVER_ABORTS,
                    payload.client_id,
                ),
                Message(
                    delay=self.network.delay(),
                    target=abort_handler,
                ),
            )
        else:
            # The write commits.
            # No need to tell the client the good news.
            # In effect, clients assume they committed when they don't hear back.
            self.version += 1
            return (
                Event(
                    EventType.SERVER_COMMITS,
                    payload.client_id,
                    f"version={self.version}",
                ),
                None,
            )


class ReadWriteClient:
    """
    Simulates a client sending read-then-write requests to a server over the network.

    It reads the version number, then tries to write, passing the version.
    If the write succeeds, it stops.
    Else, it backs off and retries.
    """

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.id!r})"

    def __init__(self, id: int, network: Network, server: ReadWriteOCCServer, backoffs: Iterator[float]):
        self.id = id
        self.network = network
        self.server = server
        self.backoffs = backoffs

    def initiate(self, payload: None, reply_target: None) -> TargetResult:
        return (
            Event(
                EventType.CLIENT_REQUESTS_VERSION,
                self.id,
            ),
            Message(
                delay=self.network.delay(),
                target=self.server.handle_read,
                payload=self.id,
                reply_target=self.handle_read_response,
            ),
        )

    def handle_read_response(self, version: int, reply_target: None) -> TargetResult:
        return (
            Event(
                EventType.CLIENT_REQUESTS_WRITE,
                self.id,
            ),
            Message(
                delay=self.network.delay(),
                target=self.server.handle_write,
                payload=MaybeCommitPayload(version, self.id),
                reply_target=self.handle_abort,
            ),
        )

    def handle_abort(self, payload: None, reply_target: None) -> TargetResult:
        return (
            Event(
                EventType.CLIENT_BACKS_OFF,
                self.id,
            ),
            Message(
                delay=next(self.backoffs),  # client-internal, so no network delay
                target=self.initiate,
            ),
        )


class WriteOnlyOCCServer:
    """
    Simulates a server receiving contending write requests, using write-only optimistic concurrency control.

    The server stores a version number.
    When it receives a request, it notes the version number and tentatively writes.
    Then it checks the version again:
        if different, it aborts
        if the same, it commits and increments the version

    The writes are variable-duration.
    (If they were fixed-duration, then a write would succeed just if no write was in progress when it arrived.
    So we'd end up with a locking server except it knowably does doomed-to-abort work.)
    """

    def __init__(self, network: Network, write_mu: float, write_sigma: float):
        self.version = 0
        self.network = network
        self.write_mu = write_mu
        self.write_sigma = write_sigma

    def write_duration(self) -> float:
        return max(0, random.gauss(self.write_mu, self.write_sigma))  # TODO: maybe abs instead of max, else 0 gets too much mass?

    def handle_write(self, client_id: int, rejection_handler: Callable) -> TargetResult:
        return (
            Event(
                EventType.SERVER_TENTATIVELY_WRITES,
                client_id,
            ),
            Message(
                delay=self.write_duration(),  # server-internal, so no network delay
                target=self.maybe_commit,
                payload=MaybeCommitPayload(version=self.version, client_id=client_id),
                reply_target=rejection_handler,
            ),
        )

    def maybe_commit(self, payload: MaybeCommitPayload, rejection_handler: Callable) -> TargetResult:
        assert self.version >= payload.version, f"version has decreased: {self.version=} < {payload.version=}"

        if self.version > payload.version:
            # another write committed in the meantime
            return (
                Event(
                    EventType.SERVER_ABORTS,
                    payload.client_id,
                ),
                Message(
                    delay=self.network.delay(),
                    target=rejection_handler,
                ),
            )
        else:
            # The write commits.
            # No need to tell the client the good news.
            # In effect, clients assume they committed when they don't hear back.
            self.version += 1
            return (
                Event(
                    EventType.SERVER_COMMITS,
                    payload.client_id,
                    f"version={self.version}",
                ),
                None,
            )


class LockingServer:
    """
    Simulates a server receiving contending write requests, using pessimistic concurrency control.

    When it receives a request:
        if available, it accepts it and becomes unavailable for a while
        if unavailable, it rejects it immediately
    """

    def __init__(self, network: Network, write_mu: float, write_sigma: float):
        self.available = True
        self.network = network
        self.write_mu = write_mu
        self.write_sigma = write_sigma

    def write_duration(self) -> float:
        return max(0, random.gauss(self.write_mu, self.write_sigma))  # TODO: maybe abs instead of max, else 0 gets too much mass?

    def handle_write(self, client_id: int, rejection_handler: Callable) -> TargetResult:
        if self.available:
            self.available = False
            # No need to tell the client the good news.
            # In effect, clients assume they were accepted when they don't hear back.
            # Also, server-internal, so no network delay.
            return (
                Event(
                    EventType.SERVER_ACCEPTS,
                    client_id,
                ),
                Message(
                    delay=self.write_duration(),
                    target=self.free,
                ),
            )
        else:
            return (
                Event(
                    EventType.SERVER_REJECTS,
                    client_id,
                ),
                Message(
                    delay=self.network.delay(),
                    target=rejection_handler,
                ),
            )

    def free(self, payload: None, reply_target: None) -> TargetResult:
        # This is internal server business, not over the network.
        self.available = True
        return (
            Event(EventType.SERVER_COMMITS),
            None,
        )


class WriteOnlyClient:
    """
    Simulates a client sending write requests to a server over the network.

    If a request succeeds, it stops.
    If a request fails, it backs off and retries.
    """

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.id!r})"

    def __init__(self, id: int, network: Network, server: LockingServer | WriteOnlyOCCServer, backoffs: Iterator[float]):
        self.id = id
        self.network = network
        self.server = server
        self.backoffs = backoffs

    def initiate(self, payload: None, reply_target: None) -> TargetResult:
        return (
            Event(
                EventType.CLIENT_REQUESTS_WRITE,
                self.id,
            ),
            Message(
                delay=self.network.delay(),
                target=self.server.handle_write,
                payload=self.id,
                reply_target=self.handle_abort,
            ),
        )

    def handle_abort(self, payload: None, reply_target: None) -> TargetResult:
        return (
            Event(
                EventType.CLIENT_BACKS_OFF,
                self.id,
            ),
            Message(
                delay=next(self.backoffs),  # client-internal, so no network delay
                target=self.initiate,
            ),
        )


class Analyzer:
    def __init__(self, history: list[tuple[float, Event]]):
        self.history = history

    def total_requests(self) -> int:
        return sum(1 for _, event in self.history if event.event_type == EventType.CLIENT_REQUESTS_WRITE)

    def duration(self) -> float:
        # The time at which all client writes committed.
        time, event = self.history[-1]
        assert event.event_type == EventType.SERVER_COMMITS, f"{event.event_type=}"
        return time


class Simulation:
    def __init__(self, server: ReadWriteOCCServer | WriteOnlyOCCServer | LockingServer, clients: list[WriteOnlyClient] | list[ReadWriteClient], label: str):
        self.server = server
        self.clients = clients
        self.label = label
        self.time = 0.0  # virtual clock
        self.todos: list[Todo] = []  # heap
        self.history: list[tuple[float, Event]] = []

    def run(self):
        for client in self.clients:
            heapq.heappush(self.todos, Todo(0.0, target=client.initiate))

        # the simulation loop
        while self.todos:
            # Get the next todo.
            todo = heapq.heappop(self.todos)

            # Advance the virtual clock to the todo's scheduled time.
            assert todo.time >= self.time, f"{todo.time=}, {self.time=}"
            self.time = todo.time

            # Do the todo, maybe scheduling another todo in consequence.
            event, message = todo.target(todo.payload, todo.reply_target)
            self.history.append((self.time, event))
            if message:
                heapq.heappush(
                    self.todos,
                    Todo(
                        self.time + message.delay,
                        message.target,
                        message.payload,
                        message.reply_target,
                    ),
                )


def expo(base: int, cap: float) -> Iterator[float]:
    return (min(cap, base * 2**n) for n in count())


def full_jitter(raw: Iterator[float]) -> Iterator[float]:
    return (random.uniform(0, t) for t in raw)


def half_jitter(raw: Iterator[float]) -> Iterator[float]:
    return (t / 2 + random.uniform(0, t / 2) for t in raw)


def normal_jitter(raw: Iterator[float], mu: float, sigma: float) -> Iterator[float]:
    return (max(0, t + random.gauss(mu, sigma)) for t in raw)


def main() -> None:
    simulations: list[Simulation] = []
    network = Network(10, 2)
    num_clients = 2

    locking_server = LockingServer(network, write_mu=5, write_sigma=1)
    locking_clients = [WriteOnlyClient(i, network, locking_server, repeat(3)) for i in range(num_clients)]
    locking_label = "locking, always 3"
    simulations.append(Simulation(locking_server, locking_clients, locking_label))

    write_only_occ_server = WriteOnlyOCCServer(network, write_mu=5, write_sigma=1)
    write_only_occ_clients = [WriteOnlyClient(i, network, write_only_occ_server, repeat(3)) for i in range(num_clients)]
    write_only_occ_label = "write only occ, always 3"
    simulations.append(Simulation(write_only_occ_server, write_only_occ_clients, write_only_occ_label))

    read_write_occ_server = ReadWriteOCCServer(network, write_mu=5, write_sigma=1)
    read_write_occ_clients = [ReadWriteClient(i, network, read_write_occ_server, repeat(3)) for i in range(num_clients)]
    read_write_occ_label = "read-write occ, always 3"
    simulations.append(Simulation(read_write_occ_server, read_write_occ_clients, read_write_occ_label))

    for sim in simulations:
        sim.run()

    for sim in simulations:
        analyzer = Analyzer(sim.history)
        print(f"== {sim.label} ==")
        print(f"requests: {analyzer.total_requests()}, duration: {analyzer.duration():.2f}")
        for time, event in analyzer.history:
            print(f"{time:.2f}: {event}")


if __name__ == "__main__":
    main()
