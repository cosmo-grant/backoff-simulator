# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///

"""
Simulates various backoff strategies for contending writes over the network.

The setup:
- many clients, starting at the same time, request over the network to write a particular database row
- if writes contend, only one commits: the rest are discarded and the clients have to retry

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


@dataclass(frozen=True)
class Message:
    """
    Message from simulated resource to simulation loop.

    If target is not None, read as:
        please schedule this: after delay, call target with payload, and tell it to reply by calling reply_target
    Else, read as:
        nothing to schedule

    Also includes a human-readable description.
    """

    description: str
    delay: float | None = None
    target: Callable | None = None
    payload: int | None = None
    reply_target: Callable | None = None

    def __post_init__(self):
        if self.target is not None and self.delay is None:
            raise ValueError(f"must have non-None delay for non-None target: {self.delay=}, {self.target=}")


@dataclass(order=True, frozen=True)
class Todo:
    """
    Scheduled unit of work for the simulation loop.

    Read as: at time, call target with payload and tell it to reply by calling reply_target.
    """

    time: float
    target: Callable = field(compare=False)
    payload: int | None = field(compare=False, default=None)
    reply_target: Callable | None = field(compare=False, default=None)


class Network:
    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma

    def delay(self) -> float:
        return max(0, random.gauss(self.mu, self.sigma))


class PCCServer:
    """
    Simulates a server receiving contending write requests, using pessimistic concurrency control.

    When it receives a request:
        if available, it accepts it and becomes unavailable for a while
        if unavailable, it rejects it immediately
    """

    def __init__(self, network: Network, busy_for: int):
        self.available = True
        self.network = network
        self.busy_for = busy_for

    def receive(self, client_id: int, rejection_handler: Callable) -> Message:
        if self.available:
            self.available = False
            # No need to tell the client the good news.
            # In effect, clients assume they were accepted when they don't hear back.
            # Also, no network delay for this todo: it's server-internal.
            message = Message(
                delay=self.busy_for,
                target=self.free,
                description=f"server accepts client {client_id}",
            )
        else:
            message = Message(
                delay=self.network.delay(),
                target=rejection_handler,
                description=f"server rejects client {client_id}",
            )

        return message

    def free(self, payload: None, reply_target: None) -> Message:
        # This is internal server business, not over the network.
        self.available = True
        return Message(description="server free")


class Client:
    """
    Simulates a client sending write requests to a server over the network.

    If a request succeeds, it stops.
    If a request fails, it backs off and retries.
    """

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.id!r})"

    def __init__(self, id: int, network: Network, server: PCCServer, backoffs: Iterator[float]):
        self.id = id
        self.network = network
        self.server = server
        self.backoffs = backoffs
        self.request_count = 0

    def send(self, payload: None, reply_target: None) -> Message:
        self.request_count += 1
        return Message(
            delay=self.network.delay(),
            target=self.server.receive,
            payload=self.id,
            reply_target=self.handle_rejection,
            description=f"client {self.id} sends",
        )

    def handle_rejection(self, payload: None, reply_target: None) -> Message:
        return Message(
            delay=next(self.backoffs),  # client-internal, so no network delay
            target=self.send,
            description=f"client {self.id} backs off",
        )


class Simulation:
    def __init__(self, busy_for: int, backoffs_factory: Callable[[], Iterator[float]], num_clients: int, label: str):
        self.time = 0.0  # virtual clock
        self.todos: list[Todo] = []  # heap
        self.history: list[tuple[float, str]] = []
        self.network = Network(10, 2)
        self.server = PCCServer(self.network, busy_for)
        self.clients = [Client(i, self.network, self.server, backoffs_factory()) for i in range(num_clients)]
        self.num_clients = num_clients
        self.label = label

    def run(self):
        for client in self.clients:
            heapq.heappush(self.todos, Todo(0.0, target=client.send))

        # the simulation loop
        while self.todos:
            # Get the next todo.
            todo = heapq.heappop(self.todos)

            # Advance the virtual clock to the todo's scheduled time.
            assert todo.time >= self.time, f"{todo.time=}, {self.time=}"
            self.time = todo.time

            # Do the todo, returning a message.
            message = todo.target(todo.payload, todo.reply_target)
            self.history.append((self.time, message.description))
            if message.target:
                # The message describes a new todo.
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
        return sum(client.request_count for client in self.clients)

    def duration(self) -> float:
        return self.time


def expo(base: int, cap: float) -> Iterator[float]:
    return (min(cap, base * 2**n) for n in count())


def full_jitter(raw: Iterator[float]) -> Iterator[float]:
    return (random.uniform(0, t) for t in raw)


def half_jitter(raw: Iterator[float]) -> Iterator[float]:
    return (t / 2 + random.uniform(0, t / 2) for t in raw)


def normal_jitter(raw: Iterator[float], mu: float, sigma: float) -> Iterator[float]:
    return (max(0, t + random.gauss(mu, sigma)) for t in raw)


def main():
    simulations = [
        Simulation(10, lambda: repeat(3), 2, "always 3"),
        # Simulation(10, lambda: full_jitter(expo(5, 200)), 20, "full jittered expo"),
        # Simulation(10, lambda: half_jitter(expo(5, 200)), 20, "half jittered expo"),
        # Simulation(10, lambda: normal_jitter(expo(5, 200), 0, 1), 20, "normal jittered expo"),
    ]
    for sim in simulations:
        sim.run()

    for sim in simulations:
        print(f"== {sim.label} ==")
        print(f"requests: {sim.total_requests()}, duration {sim.duration():.2f}")
        for time, description in sim.history:
            print(f"{time:.2f}: {description}")


if __name__ == "__main__":
    main()
