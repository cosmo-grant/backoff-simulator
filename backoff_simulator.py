import heapq
import random
from argparse import ArgumentParser
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import count, product
from typing import Callable, Iterator, Protocol

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tabulate import tabulate


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
    CLIENT_REQUESTS_WRITE = "client_requests_write"  # counting requests is counting these events
    CLIENT_BACKS_OFF = "client_backs_off"
    SERVER_COMMITS = "server_commits"  # last event is always this type

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


class BackoffStrategy(Protocol):
    def get_backoffs(self) -> Iterator[float]: ...


class Expo(BackoffStrategy):
    def __init__(self, base: int, cap: int):
        self.base = base
        self.cap = cap

    def get_backoffs(self) -> Iterator[float]:
        return (min(self.cap, self.base * 2**n) for n in count())


class FullJitteredExpo(BackoffStrategy):
    def __init__(self, base: int, cap: int):
        self.base = base
        self.cap = cap

    def get_backoffs(self) -> Iterator[float]:
        return (random.uniform(0, t) for t in Expo(self.base, self.cap).get_backoffs())


class Simulation:
    def __init__(self, server: ReadWriteOCCServer | WriteOnlyOCCServer | LockingServer, clients: list[WriteOnlyClient] | list[ReadWriteClient], backoff_strategy: BackoffStrategy):
        self.server = server
        self.clients = clients
        self.backoff_strategy = backoff_strategy  # used as metadata
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

    def total_requests(self) -> int:
        assert self.history, f"{self.history=}. Have you run the simulation?"
        return sum(1 for _, event in self.history if event.event_type == EventType.CLIENT_REQUESTS_WRITE)

    def duration(self) -> float:
        # The time at which all client writes committed.
        assert self.history, f"{self.history=}. Have you run the simulation?"
        time, event = self.history[-1]
        assert event.event_type == EventType.SERVER_COMMITS, f"{event.event_type=}"
        return time


def set_up_simulations(
    max_clients: int,
    expo_base: int,
    expo_cap: int,
    network_mu: int,
    network_sigma: int,
    write_mu: int,
    write_sigma: int,
    repeat: int,
) -> list[Simulation]:
    """
    Return ready-to-run simulations, for various (client count, backoff strategy, concurrency control) combinations.

    We repeat simulations for any given combination, so we can average out noise when analyzing the results.
    This function needs to be passed all simulation-relevant parameters.
    """
    network = Network(network_mu, network_sigma)
    concurrency_controls = [
        (LockingServer, WriteOnlyClient),
        (WriteOnlyOCCServer, WriteOnlyClient),
        (ReadWriteOCCServer, ReadWriteClient),
    ]
    backoff_strategies = [Expo(expo_base, expo_cap), FullJitteredExpo(expo_base, expo_cap)]

    simulations: list[Simulation] = []
    for backoff_strategy, (server_cls, client_cls), num_clients, i in product(
        backoff_strategies,
        concurrency_controls,
        range(1, max_clients + 1),
        range(1, repeat + 1),
    ):
        server = server_cls(network, write_mu, write_sigma)
        simulations.append(
            Simulation(
                server,
                [client_cls(j, network, server, backoff_strategy.get_backoffs()) for j in range(num_clients)],
                backoff_strategy,
            )
        )

    return simulations


def run(
    max_clients: int = 50,
    expo_base: int = 2,
    expo_cap: int = 10,
    network_mu: int = 10,
    network_sigma: int = 2,
    write_mu: int = 5,
    write_sigma: int = 2,
    repeat: int = 20,
    work_over_duration: float = 1,
) -> None:
    simulations = set_up_simulations(max_clients, expo_base, expo_cap, network_mu, network_sigma, write_mu, write_sigma, repeat)

    for sim in simulations:
        sim.run()

    # Analyze the results.
    # Group simulations by (num_clients, strategy, concurrency_control).
    groups: dict[tuple[int, str, str], list[Simulation]] = {}
    for sim in simulations:
        key = (
            len(sim.clients),
            type(sim.backoff_strategy).__name__,
            type(sim.server).__name__,
        )
        groups.setdefault(key, []).append(sim)

    # Average per group of total requests and duration.
    results: dict[tuple[int, str, str], tuple[float, float, float]] = {}
    for key, sims in groups.items():
        avg_requests = sum(s.total_requests() for s in sims) / len(sims)
        avg_duration = sum(s.duration() for s in sims) / len(sims)
        # Cost is a measure of performance (lower is better), from combining total requests and duration.
        # The work_over_duration is the exchange rate of requests to duration.
        # For example, a value of 5 means you're indifferent between 5 extra requests vs 1 extra millisecond.
        avg_cost = sum(work_over_duration * s.total_requests() + s.duration() for s in sims) / len(sims)
        results[key] = (avg_requests, avg_duration, avg_cost)

    # We run every strategy-control combination.
    controls = sorted({control for _, _, control in results})
    strategies = sorted({strategy for _, strategy, _ in results})

    # Plot 1: total requests vs number of clients for each strategy, one subplot per control
    fig1, axes1 = plt.subplots(1, len(controls), figsize=(5 * len(controls), 5), sharey=True)
    for ax, control in zip(axes1, controls):
        for strategy in strategies:
            xs = sorted(n for n, s, c in results if s == strategy and c == control)
            ys = [results[(n, strategy, control)][0] for n in xs]  # requests
            ax.plot(xs, ys, label=strategy)
        ax.set_xlabel("number of clients")
        ax.set_ylabel("total requests (avg)")
        ax.legend()
        ax.set_title(control)
    fig1.suptitle("Work")
    fig1.tight_layout()
    fig1.savefig("work.png")

    # Plot 2: duration vs number of clients for each strategy, one subplot per control
    fig2, axes2 = plt.subplots(1, len(controls), figsize=(5 * len(controls), 5), sharey=True)
    for ax, control in zip(axes2, controls):
        for strategy in strategies:
            xs = sorted(n for n, s, c in results if s == strategy and c == control)
            ys = [results[(n, strategy, control)][1] for n in xs]  # duration
            ax.plot(xs, ys, label=strategy)
        ax.set_xlabel("number of clients")
        ax.set_ylabel("duration (avg)")
        ax.legend()
        ax.set_title(control)
    fig2.suptitle("Duration")
    fig2.tight_layout()
    fig2.savefig("duration.png")

    # Plot 3: cost vs number of clients for each strategy, one subplot per control
    fig3, axes3 = plt.subplots(1, len(controls), figsize=(5 * len(controls), 5), sharey=True)
    for ax, control in zip(axes2, controls):
        for strategy in strategies:
            xs = sorted(n for n, s, c in results if s == strategy and c == control)
            ys = [results[(n, strategy, control)][2] for n in xs]  # cost
            ax.plot(xs, ys, label=strategy)
        ax.set_xlabel("number of clients")
        ax.set_ylabel("cost (avg)")
        ax.legend()
        ax.set_title(control)
    fig2.suptitle("Cost")
    fig2.tight_layout()
    fig2.savefig("cost.png")

    # Plot 4: scatter plots of write-request times.
    # One subplot per (strategy, control) combination, using an arbitrary sim at max num_clients.
    fig3, axes = plt.subplots(len(strategies), len(controls), figsize=(5 * len(controls), 5 * len(strategies)))
    for ax, (strategy, control) in zip(axes.flat, product(strategies, controls)):
        sim = groups[(max_clients, strategy, control)][0]  # pick first repetition as representative
        times = []
        client_ids = []
        for time, event in sim.history:
            if event.event_type == EventType.CLIENT_REQUESTS_WRITE:
                times.append(time)
                client_ids.append(event.client_id)
        ax.scatter(times, client_ids, s=4, alpha=0.5)
        ax.set_title(f"{strategy}\n{control}", fontsize=9)
        ax.set_xlabel("time")
        ax.set_ylabel("client id")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig3.suptitle("Write Requests Over Time")
    fig3.tight_layout()
    fig3.savefig("scatter.png")

    # Stdout: history of the simulation.
    # One history per (strategy, control) combination, using an arbitrary sim at a low num_clients.
    for strategy, control in product(strategies, controls):
        num_clients = min(max_clients, 3)
        sim = groups[(num_clients, strategy, control)][0]  # pick first repetition as representative
        print(f"{control} + {strategy}", end="\n\n")
        print(
            tabulate(
                (
                    (
                        format(t, ".02f"),
                        e.client_id if e.client_id is not None else "-",
                        e.event_type,
                        e.event_detail if e.event_detail is not None else "",
                    )
                    for t, e in sim.history
                ),
                headers=["time", "client id", "event type", "event detail"],
            )
        )
        print("\n\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--max-clients", type=int)
    parser.add_argument("--expo-base", type=int)
    parser.add_argument("--expo-cap", type=int)
    parser.add_argument("--network-mu", type=int)
    parser.add_argument("--network-sigma", type=int)
    parser.add_argument("--write-mu", type=int)
    parser.add_argument("--write-sigma", type=int)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--work-over-duration", type=float)
    args = parser.parse_args()

    # rely on simulate()'s defaults for not-provided args
    run(**{k: v for k, v in vars(args).items() if v is not None})
