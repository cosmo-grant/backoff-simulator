# Backoff Simulator

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
