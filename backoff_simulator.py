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
from typing import Callable, Iterator


@dataclass(order=True)
class ScheduledTask:
    time: float
    task: Callable = field(compare=False)


class Server:
    """
    Simulates a server receiving contending write requests over the network.

    When it receives a request:
        - if available, it accepts it and becomes unavailable for a while
        - if unavailable, it rejects it
    """

    def __init__(self, busy_for: int):
        self.available = True
        self.busy_for = busy_for

    def receive(self) -> bool:
        accepted = self.available
        self.available = False

        return accepted

    def free(self) -> None:
        self.available = True


class Client:
    """
    Simulates a client sending write requests to a server over the network.

    If a request succeeds, it stops.
    If a request fails, it backs off and retries.
    """

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.id!r})"

    def __init__(self, id: int, server: Server, backoffs: Iterator[float]):
        self.id = id
        self.server = server
        self.backoffs = backoffs
        self.request_count = 0

    def send(self) -> tuple[float, Callable]:
        self.request_count += 1
        accepted = self.server.receive()
        if accepted:
            return (self.server.busy_for, self.server.free)
        else:
            return (next(self.backoffs), self.send)


class Simulation:
    def __init__(self, busy_for: int, backoffs_factory: Callable[[], Iterator[float]], num_clients: int, label: str):
        self.time = 0.0  # virtual clock
        self.todo: list[ScheduledTask] = []  # heap
        self.server = Server(busy_for)
        self.clients = [Client(i, self.server, backoffs_factory()) for i in range(num_clients)]
        self.num_clients = num_clients
        self.label = label

    def run(self):
        for client in self.clients:
            heapq.heappush(self.todo, ScheduledTask(0.0, client.send))

        while self.todo:
            task = heapq.heappop(self.todo)
            self.time = task.time
            todo = task.task()
            if todo:
                delay, task = todo
                heapq.heappush(self.todo, ScheduledTask(self.time + delay, task))

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
    return (t + random.gauss(mu, sigma) for t in raw)


def main():
    simulations = [
        Simulation(1, lambda: repeat(3), 2, "always 3"),
        # Simulation(1, lambda: full_jitter(expo(2, 10)), 20, "full jittered expo"),
    ]
    for sim in simulations:
        sim.run()

    for sim in simulations:
        print(f"{sim.label}: {sim.total_requests()} requests, {sim.duration():.2f}s duration")


if __name__ == "__main__":
    main()
