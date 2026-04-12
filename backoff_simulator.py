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
You can keep the request count low by making clients spread out their tries.
But then the completion time is high.
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
from typing import Callable, Iterator, Any


@dataclass(frozen=True)
class Todo:
    """
    Interpret as: after delay, call target with payload
    and tell it to reply by calling reply_target,
    plus a description for posterity.
    """

    delay: float
    target: Callable
    payload: Any
    reply_target: Callable
    description: str


@dataclass(order=True, frozen=True)
class ScheduledTodo:
    """When to do what, processed by the simulation loop."""

    time: float
    todo: Todo = field(compare=False)


class Network:
    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma

    def delay(self) -> float:
        return max(0, random.gauss(self.mu, self.sigma))


class PCCServer:
    """
    Simulates a server receiving write requests, using pessimistic concurrency control.

    When it receives a request:
        if available, it accepts it and becomes unavailable for a while
        if unavailable, it rejects it immediately
    """

    def __init__(self, network: Network, busy_for: int):
        self.available = True
        self.network = Network
        self.busy_for = busy_for

    def receive(self, client_id: int, rejection_handler: Callable) -> Todo:
        if self.available:
            self.available = False
            # No need to tell the client the good news.
            # In effect, clients assume they were accepted when they don't hear back.
            # Also, no network delay for this todo: it's server-internal.
            todo = Todo(
                delay=self.server.busy_for,
                target=self.server.free,
                payload=None,
                reply_to=None,
                description="server free",
            )
        else:
            todo = Todo(
                delay=self.network.delay(),
                target=rejection_handler,
                payload=None,
                reply_to=None,
                description=f"client {client_id} backs off",
            )

        return todo

    def free(self, payload: None, reply_target: None) -> None:
        # This is internal server business, not over the network.
        self.available = True


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

    def send(self, payload: None, reply_target: None) -> Todo:
        self.request_count += 1
        return Todo(
            delay=self.network.delay(),
            target=self.server.receive,
            payload=self.id,
            reply_target=self.handle_rejection,
            description=f"client {self.id} received",
        )

    def handle_rejection(self, payload: None, reply_target: None) -> Todo:
        return Todo(
            delay=next(self.backoffs),  # client-internal, so no network delay
            target=self.send,
            payload=None,
            reply_target=None,
            description=f"client {self.id} sent",
        )


class Simulation:
    def __init__(self, busy_for: int, backoffs_factory: Callable[[], Iterator[float]], num_clients: int, label: str):
        self.time = 0.0  # virtual clock
        self.scheduled_todos: list[ScheduledTodo] = []  # heap
        self.history: list[tuple[float, str]] = []
        self.network = Network(10, 2)
        self.server = PCCServer(self.network, busy_for)
        self.clients = [Client(i, self.network, self.server, backoffs_factory()) for i in range(num_clients)]
        self.num_clients = num_clients
        self.label = label

    def run(self):
        for client in self.clients:
            heapq.heappush(
                self.scheduled_todos,
                ScheduledTodo(
                    0.0,
                    Todo(
                        delay=None,  # TODO: make defaults?
                        target=client.send,
                        payload=None,
                        reply_target=None,
                        description=f"client {client.id} sent",
                    ),
                ),
            )

        # the simulation loop
        while self.scheduled_todos:
            todo_now = heapq.heappop(self.scheduled_todos)

            assert todo_now.time > self.time, f"{todo_now.time=}, {self.time=}"
            self.time = todo_now.time

            self.history.append((self.time, todo_now.description))

            todo_later = todo_now.target(todo_now.payload, todo_now.reply_target)
            if todo_later:
                heapq.heappush(self.scheduled_todos, ScheduledTodo(self.time + todo_later.delay, todo_later))

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
        Simulation(10, lambda: full_jitter(expo(5, 200)), 20, "full jittered expo"),
        Simulation(10, lambda: half_jitter(expo(5, 200)), 20, "half jittered expo"),
        Simulation(10, lambda: normal_jitter(expo(5, 200), 0, 1), 20, "normal jittered expo"),
    ]
    for sim in simulations:
        sim.run()

    for sim in simulations:
        print(f"== {sim.label} ==")
        print(f"{sim.total_requests()} requests, {sim.duration():.2f}s duration")
        for time, description in sim.history:
            print(f"{time:.2f}: {description}")


if __name__ == "__main__":
    main()
