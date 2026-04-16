import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Backoff Simulator

    Many clients request over the network that the server write a particular value.
    If writes contend, only one commits.
    Clients back off and retry until their write commits.

    You want to keep low:
    - the time until all writes commit (**duration**)
    - the total number of requests (**work**)

    You can keep the duration low by making clients retry rapidly.
    But then writes often contend, so the work is high.

    You can keep the work low by making clients retry sporadically.
    But then the server is often idle, so the duration is high.

    So there’s a **tradeoff**.

    The **cost** is a combined measure of duration and work.
    It's set by a work-to-duration exchange rate, say
    - 1 if you think 1 extra request and 1 extra ms are equally bad
    - or 5 if you think 5 extra requests and 1 extra ms are equally bad
    - or 0.2 if you think 1 extra request and 5 extra ms are equally bad.

    There's a famous aws blog post and simulation script about this:
    - https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter
    - https://github.com/aws-samples/aws-arch-backoff-simulator

    This app is based on those.
    But I re-implemented the simulation (til about heap-based priority queues) and added a few bells and whistles.

    The blog post is clear:

    > The return on implementation complexity of using jittered backoff is huge,
    and it should be considered a standard approach for remote clients.

    But is that true of **your use case**?

    **Let's explore.**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simulation
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    import heapq
    import random
    from collections.abc import Callable, Iterator
    from dataclasses import dataclass, field
    from enum import StrEnum
    from itertools import count, product, repeat
    from typing import Protocol

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
            server: ReadWriteOCCServer | WriteOnlyOCCServer | LockingServer,
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
            # The time at which all client writes committed.
            assert self.history, f"{self.history=}. Have you run the simulation?"
            time, event = self.history[-1]
            assert event.event_type == EventType.SERVER_COMMITS, f"{event.event_type=}"
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


    def set_up_simulations(
        max_clients: int,
        constant: float,
        expo_base: float,
        expo_cap: float,
        network_mu: float,
        network_sigma: float,
        write_mu: float,
        write_sigma: float,
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
        backoff_strategies = [
            Constant(constant),
            Expo(expo_base, expo_cap),
            FullJitteredExpo(expo_base, expo_cap),
            EqualJitteredExpo(expo_base, expo_cap),
        ]

        simulations: list[Simulation] = []
        for backoff_strategy, (server_cls, client_cls), num_clients, _ in product(
            backoff_strategies,
            concurrency_controls,
            get_client_nums(max_clients),
            range(repeat),
        ):
            server = server_cls(network, write_mu, write_sigma)
            simulations.append(
                Simulation(
                    server,
                    [client_cls(j, network, server, backoff_strategy.get_backoffs()) for j in range(num_clients)],  # ty: ignore[invalid-argument-type]  # checker can't see server/client type correlation
                    backoff_strategy,
                )
            )

        return simulations


    type SimulationType = tuple[int, str, str]  # number of clients, backoff strategy, concurrency control
    type SimulationGroups = dict[SimulationType, list[Simulation]]


    @dataclass(frozen=True)
    class Metrics:
        requests: float
        duration: float
        cost: float


    type SimulationResults = dict[SimulationType, Metrics]


    def simulate(
        max_clients: int,
        constant: float,
        expo_base: float,
        expo_cap: float,
        network_mu: float,
        network_sigma: float,
        write_mu: float,
        write_sigma: float,
        repeat: int,
        work_to_duration: float,
    ) -> SimulationGroups:
        """Run simulations and return them, grouped by type."""
        simulations = set_up_simulations(max_clients, constant, expo_base, expo_cap, network_mu, network_sigma, write_mu, write_sigma, repeat)

        for sim in simulations:
            sim.run()

        groups: SimulationGroups = {}
        for sim in simulations:
            key = (
                len(sim.clients),
                type(sim.backoff_strategy).__name__,
                type(sim.server).__name__,
            )
            groups.setdefault(key, []).append(sim)

        return groups


    def make_figures(groups: SimulationGroups, max_clients: int, work_to_duration: float = 1) -> dict[str, list[tuple[str, plt.Figure]]]:
        """Create and return the analysis figures (without saving to disk)."""

        # Average per group of requests, duration, and cost.
        results: SimulationResults = {}
        for key, sims in groups.items():
            avg_requests = sum(s.work() for s in sims) / len(sims)
            avg_duration = sum(s.duration() for s in sims) / len(sims)
            # Cost is a measure of performance (lower is better), from combining total requests and duration.
            # The work_to_duration is the exchange rate of requests to duration.
            # For example, a value of 5 means you're indifferent between 5 extra requests vs 1 extra millisecond.
            avg_cost = sum(work_to_duration * s.work() + s.duration() for s in sims) / len(sims)
            results[key] = Metrics(avg_requests, avg_duration, avg_cost)

        controls = sorted({control for _, _, control in results})
        strategies = sorted({strategy for _, strategy, _ in results})

        figures: dict[str, list[tuple[str, plt.Figure]]] = {}

        metric_specs = [
            ("total requests (avg)", "requests"),
            ("duration (avg)", "duration"),
            ("cost (avg)", "cost"),
        ]

        for control in controls:
            # Metrics figure, one subplot per metric
            fig_m, axes_m = plt.subplots(1, len(metric_specs), figsize=(5 * len(metric_specs), 5))
            for ax, (ylabel, attr) in zip(axes_m, metric_specs, strict=True):
                for strategy in strategies:
                    xs = get_client_nums(max_clients)
                    ys = [getattr(results[(n, strategy, control)], attr) for n in xs]
                    ax.plot(xs, ys, label=strategy)
                ax.set_xlabel("number of clients")
                ax.set_ylabel(ylabel)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.legend()
            fig_m.suptitle(control)
            fig_m.tight_layout()
            figures.setdefault("metrics", []).append((control, fig_m))

            # Scatter figure, one subplot per strategy in a 2-column grid
            nrows = (len(strategies) + 1) // 2
            fig_s, axes_s = plt.subplots(nrows, 2, squeeze=False, figsize=(10, 5 * nrows))
            axes_flat = axes_s.flatten()
            for ax in axes_flat[len(strategies) :]:
                ax.set_visible(False)
            for ax, strategy in zip(axes_flat, strategies, strict=False):
                sim = groups[(max_clients, strategy, control)][0]
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
            fig_s.suptitle(control)
            fig_s.tight_layout()
            figures.setdefault("scatter", []).append((control, fig_s))

        return figures


    def make_tables(
        groups: SimulationGroups,
        max_clients: int,
    ) -> dict[tuple[str, str], str]:
        """Return the simulation history tables as a string."""

        controls = sorted({control for _, _, control in groups})
        strategies = sorted({strategy for _, strategy, _ in groups})

        client_nums = sorted(get_client_nums(max_clients))
        smallest_interesting = next(
            (n for n in client_nums if n > 2),
            2 if 2 in client_nums else 1,
        )

        tables: dict[tuple[str, str], str] = {}
        for strategy, control in product(strategies, controls):
            sim = groups[(smallest_interesting, strategy, control)][0]  # pick first repetition as representative
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
            tables[(strategy, control)] = table

        return tables


    def run(
        max_clients: int,
        constant: float,
        expo_base: float,
        expo_cap: float,
        network_mu: float,
        network_sigma: float,
        write_mu: float,
        write_sigma: float,
        repeat: int,
        work_to_duration: float,
    ) -> None:
        groups = simulate(
            max_clients,
            constant,
            expo_base,
            expo_cap,
            network_mu,
            network_sigma,
            write_mu,
            write_sigma,
            repeat,
            work_to_duration,
        )

        figs = make_figures(groups, max_clients, work_to_duration)
        for kind, figs_ in figs.items():
            for control, fig in figs_:
                fig.savefig(f"{control}_{kind}.png")

        tables = make_tables(groups, max_clients)
        for (strategy, control), table in tables.items():
            print(f"{control} + {strategy}\n\n{table}\n")

    return make_figures, make_tables, simulate


@app.cell
def _(mo):
    form = (
        mo.Html("""
        <style>
            .params-table td {{ padding: 4px 80px; }}
        </style>
        <table class="params-table">
            <tr><td><strong>max clients</strong></td><td>{max_clients}</td></tr>
            <tr><td><strong>constant</strong></td><td>{constant}</td></tr>
            <tr><td><strong>expo base</strong></td><td>{expo_base}</td></tr>
            <tr><td><strong>expo cap</strong></td><td>{expo_cap}</td></tr>
            <tr><td><strong>network mu</strong></td><td>{network_mu}</td></tr>
            <tr><td><strong>network sigma</strong></td><td>{network_sigma}</td></tr>
            <tr><td><strong>write mu</strong></td><td>{write_mu}</td></tr>
            <tr><td><strong>write sigma</strong></td><td>{write_sigma}</td></tr>
            <tr><td><strong>repeat</strong></td><td>{repeat}</td></tr>
            <tr><td><strong>work to duration</strong></td><td>{work_to_duration}</td></tr>
        </table>
        """)
        .batch(
            max_clients=mo.ui.number(start=1, stop=100, step=1, value=50),
            constant=mo.ui.number(start=0.0, stop=10, step=0.1, value=0.5),
            expo_base=mo.ui.number(start=0.0, stop=10, step=0.1, value=2),
            expo_cap=mo.ui.number(start=0.0, stop=5000, step=100, value=1000),
            network_mu=mo.ui.number(start=0.0, stop=50, step=0.1, value=10),
            network_sigma=mo.ui.number(start=0.0, stop=10, step=0.1, value=2),
            write_mu=mo.ui.number(start=0.0, stop=50, step=0.1, value=2),
            write_sigma=mo.ui.number(start=0.0, stop=10, step=0.1, value=1),
            repeat=mo.ui.number(start=1, stop=100, step=1, value=20),
            work_to_duration=mo.ui.number(start=0.1, stop=10, step=0.1, value=1),
        )
        .form()
    )

    form  # noqa: B018
    return (form,)


@app.cell
def _(form, make_figures, make_tables, mo, simulate):
    mo.stop(form.value is None)
    params = form.value
    groups = simulate(**params)
    figs = make_figures(groups, params["max_clients"], params["work_to_duration"])
    tables = make_tables(groups, params["max_clients"])
    return figs, params, tables


@app.cell(hide_code=True)
def _(form, mo, params):
    mo.stop(form.value is None)
    mo.md(rf"""
        ### Work, Duration, Cost 

        Each metric is an average across the {params["repeat"]} repetitions.
    """)
    return


@app.cell
def _(figs, mo):
    metrics_tabs = mo.ui.tabs({control: mo.as_html(fig) for control, fig in figs["metrics"]})
    metrics_tabs  # noqa: B018
    return


@app.cell(hide_code=True)
def _(form, mo, params):
    mo.stop(form.value is None)
    mo.md(rf"""
        ### Write Requests Over Time

        Each scatter plot records a simulation with {params["max_clients"]} clients. 
        A dot at $(t, i)$ means client $i$ requested a write at time $t$.
    """)
    return


@app.cell
def _(figs, mo):
    scatter_tabs = mo.ui.tabs({control: mo.as_html(fig) for control, fig in figs["scatter"]})
    scatter_tabs  # noqa: B018
    return


@app.cell(hide_code=True)
def _(form, mo, params):
    mo.stop(form.value is None)
    mo.md(rf"""
        ### Log With {min(3, params["max_clients"])} Clients
    """)
    return


@app.cell
def _(mo, tables):
    options = {f"{control} + {strategy}": mo.md(f"```\n{table}\n```") for (control, strategy), table in tables.items()}
    dropdown = mo.ui.dropdown(list(options), value=list(options)[0])
    return dropdown, options


@app.cell
def _(dropdown, mo, options):
    mo.vstack([dropdown, options[dropdown.value]])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Contention models

    How do writes contend?
    I focus on the following concurrency controls.

    **Locking server:**
    - When it receives a request:
      - if available, it accepts it and becomes unavailable for a while
      - if unavailable, it rejects it immediately.

    **Read-only OCC server:**
    - The server stores a version number.
    - When it receives a request, it notes the version number and tentatively writes.
    - Then it checks the version again:
      - if different, it aborts
      - if the same, it commits and increments the version
    - The writes are variable-duration.
      (Else a write would succeed just if no write was in progress when it arrived.
      So we'd end up with a locking server except it knowably does doomed-to-abort work.)

    **Read-write  OCC server:**
    - The server stores a version number.
    - A client first reads the version, then requests a write, passing the version.
    - The server tentatively writes.
    - Then it checks the version again:
      - if different, it aborts
      - if the same, it commits and increments the version
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Backoff strategies

    How do clients back off?
    I focus on the following strategies, where $n$ is the attempt number.

    **Constant:**
    - $c$ for all $n$
    - this relies entirely on network and write variance to spread out requests

    **Capped exponential backoff:**
    - $min(c, b \cdot 2^n)$
    - e.g. for $c=10, b=2$ this gives $2, 4, 8, 10, 10, ...$

    **Full jittered exponential backoff:**
    - $U(0, min(c, b \times 2^n))$
    - i.e. a value picked uniformly at random between 0 and the capped exponential backoff

    **Equal jittered exponential backoff:**
    - $min(c, b \times 2^n)) / 2 + U(0, min(c, b \times 2^n) / 2)$
    - i.e. half the capped exponential backoff, plus a value picked uniformly
      at random between 0 and half the capped exponential backoff
    """)
    return


if __name__ == "__main__":
    app.run()
