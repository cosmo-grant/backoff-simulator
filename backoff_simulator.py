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

import random

import asyncio
import time
from itertools import count, repeat
from typing import Iterator


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
        self.events: list[tuple[float, str]] = []

    async def receive(self, id: int) -> bool:
        if self.available:
            self.events.append((time.monotonic(), f"accepted client {id}"))
            self.available = False
            await asyncio.sleep(self.busy_for)
            self.available = True
            return True
        else:
            self.events.append((time.monotonic(), f"rejected client {id}"))
            return False


class Client:
    """
    Simulates a client sending write requests to a server over the network.

    If a request succeeds, it stops.
    If a request fails, it backs off and retries.
    """

    ids = count()

    def __init__(self, server: Server, backoffs: Iterator[float]):
        self.id = next(self.ids)
        self.server = server
        self.backoffs = backoffs
        self.request_count = 0

    async def send(self):
        accepted = False
        while True:
            accepted = await self.server.receive(self.id)
            self.request_count += 1
            if accepted:
                break
            await asyncio.sleep(next(self.backoffs))


class Simulation:
    def __init__(self, busy_for: int, backoffs: Iterator[float], num_clients: int, label: str):
        self.server = Server(busy_for)
        self.backoffs = backoffs
        self.clients = [Client(self.server, backoffs) for _ in range(num_clients)]
        self.num_clients = num_clients
        self.label = label
        self.start_time: float
        self.end_time: float

    async def run(self):
        self.start_time = time.monotonic()
        await asyncio.gather(*(client.send() for client in self.clients))
        self.end_time = time.monotonic()

    def total_requests(self) -> int:
        return sum(client.request_count for client in self.clients)

    def duration(self) -> float:
        return self.end_time - self.start_time


def expo(base: int, cap: float) -> Iterator[float]:
    return (min(cap, base * 2**n) for n in count())


def full_jitter(raw: Iterator[float]) -> Iterator[float]:
    return (random.uniform(0, t) for t in raw)


def half_jitter(raw: Iterator[float]) -> Iterator[float]:
    return (t / 2 + random.uniform(0, t / 2) for t in raw)


def normal_jitter(raw: Iterator[float], mu: float, sigma: float) -> Iterator[float]:
    return (t + random.gauss(mu, sigma) for t in raw)


async def main():
    simulations = [
        Simulation(1, repeat(3), 2, "always 3"),
        Simulation(1, full_jitter(expo(2, 10)), 20, "full jittered expo"),
        # Simulation(1, half_jitter(expo(2, 10)), 50, "half jittered expo"),
        # Simulation(1, normal_jitter(expo(2, 10), 0, 1), 50, "normal jittered expo"),
    ]
    for sim in simulations:
        await sim.run()

    for sim in simulations:
        print(f"{sim.label}: {sim.total_requests()} requests in {sim.duration():.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
