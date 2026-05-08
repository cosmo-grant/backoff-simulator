from __future__ import annotations

import heapq
import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import count, product, repeat
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .config import Spec

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tabulate import tabulate


@dataclass(frozen=True)
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
    CLIENT_REQUESTS_WRITE = "client_requests_write"  # counting requests is counting these events
    CLIENT_BACKS_OFF = "client_backs_off"

    # These happen in some controls but not others.
    CLIENT_REQUESTS_VERSION = "client_requests_version"
    SERVER_ACCEPTS = "server_accepts"
    SERVER_REJECTS = "server_rejects"
    SERVER_REPORTS_VERSION = "server_reports_version"
    SERVER_TENTATIVELY_WRITES = "server_tentatively_writes"
    SERVER_ABORTS = "server_aborts"
    SERVER_DECREMENTS = "server_decrements"  # last event for throttling server
    SERVER_COMMITS = "server_commits"  # last event for other servers


@dataclass(frozen=True)
class Event:
    """
    A simulation produces a timed sequence of events: everything of interest that happened, for later analysis.
    """

    event_type: EventType
    client_id: int
    event_detail: str = ""  # human-readable, free-form


type TargetResult = tuple[Event, Message | None]


@dataclass(order=True, frozen=True)
class Todo:
    """
    Scheduled unit of work for the simulation loop.

    Read as: "at time, call target with payload and tell it to reply by calling reply_target."
    """

    time: float
    target: Callable[..., TargetResult] = field(compare=False)
    payload: int | MaybeCommitPayload | None = field(compare=False, default=None)
    reply_target: Callable | None = field(compare=False, default=None)


class Network:
    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma

    def delay(self) -> float:
        return max(0, random.gauss(self.mu, self.sigma))


class ThrottlingServer:
    """
    Simulates a server receiving writes, throttling any over its limit in a sliding window.
    """

    def __init__(self, network: Network, limit: int, window: float):
        self.network = network
        self.limit = limit
        self.count = 0
        self.window = window

    def handle_write(self, client_id: int, rejection_handler: Callable) -> TargetResult:
        assert 0 <= self.count <= self.limit, f"{self.count=}"
        if self.count == self.limit:
            return (
                Event(EventType.SERVER_REJECTS, client_id),
                Message(delay=self.network.delay(), target=rejection_handler),
            )
        else:
            self.count += 1
            # No need to tell the client the good news.
            # In effect, clients assume they were accepted when they don't hear back.
            # Also, server-internal, so no network delay.
            return (
                Event(EventType.SERVER_ACCEPTS, client_id, f"count={self.count}"),
                Message(delay=self.window, target=self.decrement, payload=client_id),
            )

    def decrement(self, client_id: int, reply_target: None) -> TargetResult:
        # This is internal server business, not over the network.
        self.count -= 1
        return (Event(EventType.SERVER_DECREMENTS, client_id, f"count={self.count}"), None)


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
        return abs(random.gauss(self.write_mu, self.write_sigma))

    def handle_read(self, client_id: int, read_response_handler: Callable) -> TargetResult:
        return (
            Event(EventType.SERVER_REPORTS_VERSION, client_id, f"version={self.version}"),
            Message(delay=self.network.delay(), target=read_response_handler, payload=self.version),
        )

    def handle_write(self, payload: MaybeCommitPayload, abort_handler: Callable) -> TargetResult:
        return (
            Event(EventType.SERVER_TENTATIVELY_WRITES, payload.client_id),
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
                Event(EventType.SERVER_ABORTS, payload.client_id),
                Message(delay=self.network.delay(), target=abort_handler),
            )
        else:
            # The write commits.
            # No need to tell the client the good news.
            # In effect, clients assume they committed when they don't hear back.
            self.version += 1
            return (
                Event(EventType.SERVER_COMMITS, payload.client_id, f"version={self.version}"),
                None,
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
        return abs(random.gauss(self.write_mu, self.write_sigma))

    def handle_write(self, client_id: int, rejection_handler: Callable) -> TargetResult:
        return (
            Event(EventType.SERVER_TENTATIVELY_WRITES, client_id),
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
                Event(EventType.SERVER_ABORTS, payload.client_id),
                Message(delay=self.network.delay(), target=rejection_handler),
            )
        else:
            # The write commits.
            # No need to tell the client the good news.
            # In effect, clients assume they committed when they don't hear back.
            self.version += 1
            return (
                Event(EventType.SERVER_COMMITS, payload.client_id, f"version={self.version}"),
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
        return abs(random.gauss(self.write_mu, self.write_sigma))

    def handle_write(self, client_id: int, rejection_handler: Callable) -> TargetResult:
        if self.available:
            self.available = False
            # No need to tell the client the good news.
            # In effect, clients assume they were accepted when they don't hear back.
            # Also, server-internal, so no network delay.
            return (
                Event(EventType.SERVER_ACCEPTS, client_id),
                Message(delay=self.write_duration(), target=self.free, payload=client_id),
            )
        else:
            return (
                Event(EventType.SERVER_REJECTS, client_id),
                Message(delay=self.network.delay(), target=rejection_handler),
            )

    def free(self, client_id: int, reply_target: None) -> TargetResult:
        # This is internal server business, not over the network.
        self.available = True
        return (
            Event(EventType.SERVER_COMMITS, client_id),
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
            Event(EventType.CLIENT_REQUESTS_WRITE, self.id),
            Message(delay=self.network.delay(), target=self.server.handle_write, payload=self.id, reply_target=self.handle_abort),
        )

    def handle_abort(self, payload: None, reply_target: None) -> TargetResult:
        return (
            Event(EventType.CLIENT_BACKS_OFF, self.id),
            Message(
                delay=next(self.backoffs),  # client-internal, so no network delay
                target=self.initiate,
            ),
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
            Event(EventType.CLIENT_REQUESTS_VERSION, self.id),
            Message(delay=self.network.delay(), target=self.server.handle_read, payload=self.id, reply_target=self.handle_read_response),
        )

    def handle_read_response(self, version: int, reply_target: None) -> TargetResult:
        return (
            Event(EventType.CLIENT_REQUESTS_WRITE, self.id),
            Message(
                delay=self.network.delay(),
                target=self.server.handle_write,
                payload=MaybeCommitPayload(version, self.id),
                reply_target=self.handle_abort,
            ),
        )

    def handle_abort(self, payload: None, reply_target: None) -> TargetResult:
        return (
            Event(EventType.CLIENT_BACKS_OFF, self.id),
            Message(
                delay=next(self.backoffs),  # client-internal, so no network delay
                target=self.initiate,
            ),
        )


class BackoffStrategy(Protocol):
    def get_backoffs(self) -> Iterator[float]: ...


class Constant(BackoffStrategy):
    def __init__(self, constant: float):
        self.constant = constant

    def get_backoffs(self) -> Iterator[float]:
        return repeat(self.constant)


class Expo(BackoffStrategy):
    def __init__(self, base: float, cap: float):
        self.base = base
        self.cap = cap

    def get_backoffs(self) -> Iterator[float]:
        return (min(self.cap, self.base * 2**n) for n in count())


class FullJitteredExpo(BackoffStrategy):
    def __init__(self, base: float, cap: float):
        self.base = base
        self.cap = cap

    def get_backoffs(self) -> Iterator[float]:
        return (random.uniform(0, t) for t in Expo(self.base, self.cap).get_backoffs())


class EqualJitteredExpo(BackoffStrategy):
    def __init__(self, base: float, cap: float):
        self.base = base
        self.cap = cap

    def get_backoffs(self) -> Iterator[float]:
        return (t / 2 + random.uniform(0, t / 2) for t in Expo(self.base, self.cap).get_backoffs())


class Simulation:
    def __init__(
        self,
        server: ReadWriteOCCServer | WriteOnlyOCCServer | LockingServer | ThrottlingServer,
        clients: list[WriteOnlyClient] | list[ReadWriteClient],
        backoff_strategy: BackoffStrategy,
    ):
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

    def work(self) -> int:
        """Get the total number of client write requests."""
        assert self.history, f"{self.history=}. Have you run the simulation?"
        return sum(1 for _, event in self.history if event.event_type == EventType.CLIENT_REQUESTS_WRITE)

    def duration(self) -> float:
        # The time at which all client requests have been fully dealt with.
        assert self.history, f"{self.history=}. Have you run the simulation?"
        time, event = self.history[-1]
        assert event.event_type in (EventType.SERVER_COMMITS, EventType.SERVER_DECREMENTS), f"{event.event_type=}"
        return time


def get_client_nums(max_clients: int) -> list[int]:
    """
    Return a list of 1, ..., max_clients with a suitable step.

    Suppose max_clients is 100.
    It's slow and wasteful to run simulations for 1, 2, ..., 100 clients.
    Better to pick a step, e.g. 5, and run for 1, 6, 11, ..., 96, 100.
    Always includes 1 and max_clients.
    """

    max_values = 20
    if max_clients <= max_values:
        return list(range(1, max_clients + 1))
    else:
        step = (max_clients - 1) / (max_values - 1)
        return [round(1 + i * step) for i in range(max_values)]


def set_up_simulations(spec: Spec) -> list[Simulation]:
    """
    Return ready-to-run simulations given a spec.
    """
    network = Network(spec.network_mu, spec.network_sigma)

    server_factories = {
        "ThrottlingServer": (
            lambda net, params: ThrottlingServer(net, params["limit"], params["window"]),
            WriteOnlyClient,
        ),
        "LockingServer": (
            lambda net, params: LockingServer(net, params["write_mu"], params["write_sigma"]),
            WriteOnlyClient,
        ),
        "WriteOnlyOCCServer": (
            lambda net, params: WriteOnlyOCCServer(net, params["write_mu"], params["write_sigma"]),
            WriteOnlyClient,
        ),
        "ReadWriteOCCServer": (
            lambda net, params: ReadWriteOCCServer(net, params["write_mu"], params["write_sigma"]),
            ReadWriteClient,
        ),
    }

    strategy_factories = {
        "Constant": lambda params: Constant(**params),
        "Expo": lambda params: Expo(**params),
        "FullJitteredExpo": lambda params: FullJitteredExpo(**params),
        "EqualJitteredExpo": lambda params: EqualJitteredExpo(**params),
    }

    server_factory_fn, client_cls = server_factories[spec.control]
    backoff_strategies = [strategy_factories[s.type](s.params) for s in spec.strategies]

    simulations: list[Simulation] = []
    for backoff_strategy, num_clients, _ in product(
        backoff_strategies,
        get_client_nums(spec.max_clients),
        range(spec.repeat),
    ):
        server = server_factory_fn(network, spec.control_params)
        simulations.append(
            Simulation(
                server,
                [client_cls(j, network, server, backoff_strategy.get_backoffs()) for j in range(num_clients)],  # ty:ignore[invalid-argument-type]
                backoff_strategy,
            )
        )

    return simulations


type SimType = tuple[int, str]  # (num_clients, strategy_name)
type SimGroups = dict[SimType, list[Simulation]]  # the list has repeat-many simulations
type SimResults = dict[SimType, Metrics]


@dataclass(frozen=True)
class Metrics:
    requests: float
    duration: float
    cost: float


def simulate(spec: Spec) -> SimGroups:
    """Run all simulations for a spec and return them grouped by (num_clients, strategy_name)."""
    simulations = set_up_simulations(spec)
    for sim in simulations:
        sim.run()

    groups: SimGroups = {}
    for sim in simulations:
        key = (len(sim.clients), type(sim.backoff_strategy).__name__)
        groups.setdefault(key, []).append(sim)

    return groups


def make_figures(groups: SimGroups, spec: Spec) -> dict[str, plt.Figure]:
    """Create metrics and scatter figures for a spec."""

    # Compute average metrics per (num_clients, strategy).
    results: dict[SimType, Metrics] = {}
    for key, sims in groups.items():
        avg_requests = sum(s.work() for s in sims) / len(sims)
        avg_duration = sum(s.duration() for s in sims) / len(sims)
        avg_cost = sum(spec.work_to_duration * s.work() + s.duration() for s in sims) / len(sims)
        results[key] = Metrics(avg_requests, avg_duration, avg_cost)

    strategies = sorted({strategy for _, strategy in results})

    # Metrics figure: one subplot per metric, one line per strategy.
    metric_specs = [
        ("total requests (avg)", "requests"),
        ("duration (avg)", "duration"),
        ("cost (avg)", "cost"),
    ]
    fig_m, axes_m = plt.subplots(1, len(metric_specs), figsize=(5 * len(metric_specs), 5))
    for ax, (ylabel, attr) in zip(axes_m, metric_specs, strict=True):
        for strategy in strategies:
            xs = get_client_nums(spec.max_clients)
            ys = [getattr(results[(n, strategy)], attr) for n in xs]
            ax.plot(xs, ys, label=strategy)
        ax.set_xlabel("number of clients")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()
    fig_m.suptitle(spec.control)
    fig_m.tight_layout()

    # Scatter figure: one subplot per strategy.
    nrows = (len(strategies) + 1) // 2
    fig_s, axes_s = plt.subplots(nrows, 2, squeeze=False, figsize=(10, 5 * nrows))
    axes_flat = axes_s.flatten()
    for ax in axes_flat[len(strategies) :]:
        ax.set_visible(False)
    for ax, strategy in zip(axes_flat, strategies, strict=False):
        sim = groups[(spec.max_clients, strategy)][0]
        times = []
        client_ids = []
        for time, event in sim.history:
            if event.event_type == EventType.CLIENT_REQUESTS_WRITE:
                times.append(time)
                client_ids.append(event.client_id)
        ax.scatter(times, client_ids, s=4, alpha=0.5)
        ax.set_title(strategy, fontsize=9)
        ax.set_xlabel("time")
        ax.set_ylabel("client id")
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)
    fig_s.suptitle(spec.control)
    fig_s.tight_layout()

    return {"metrics": fig_m, "scatter": fig_s}


def make_tables(groups: SimGroups, spec: Spec) -> dict[str, str]:
    """Return simulation history tables (one per strategy) for a spec."""
    strategies = sorted({strategy for _, strategy in groups})

    client_nums = sorted(get_client_nums(spec.max_clients))
    smallest_interesting = next(
        (n for n in client_nums if n > 2),
        2 if 2 in client_nums else 1,
    )

    tables: dict[str, str] = {}
    for strategy in strategies:
        # pick an arbitrary (the first) repetition from the smallest interesting client count
        sim = groups[(smallest_interesting, strategy)][0]
        table = tabulate(
            (
                (
                    format(t, ".02f"),
                    e.client_id,
                    e.event_type,
                    e.event_detail,
                )
                for t, e in sim.history
            ),
            headers=["time", "client_id", "event_type", "event_detail"],
        )
        tables[strategy] = table

    return tables
