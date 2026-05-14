# Backoff Simulator

## Context

Many clients concurrently send requests over the network to a server.
The server accepts or rejects requests, according to its concurrency control.
Each client backs off and retries until its request is accepted.

You want to keep low:
- the time until all requests are accepted (_duration_)
- the total number of requests (_work_)

You can keep the duration low by making clients retry rapidly.
But then the server often rejects requests, so the work is high.

You can keep the work low by making clients retry sporadically.
But then the server is often idle, so the duration is high.

Tradeoff!

The _cost_ is an overall performance measure, got via an exchange rate: `work_to_duration * work + duration`.

Which backoff strategy minimizes the cost?

There's a well-known aws [blog post](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter) and [simulation](https://github.com/aws-samples/aws-arch-backoff-simulator) about this, focusing on optimistic concurrency control.
This app is based on those.
But I re-implemented the simulation and added more controls.

The blog post concludes:

> The return on implementation complexity of using jittered backoff is huge, and it should be considered a standard approach for remote clients.

The post has been influential.
For example, the widely-used Python package [backoff](https://github.com/litl/backoff) defaults to full jitter, citing the post.
But is it really the best strategy for _your_ use case?

Let's explore.

## How to use

Write a `simulations.toml` describing the simulations you want to run, e.g.

```toml
  [[simulation]]
  title = "Locking_Example"  # anything you like, but must be unique across [[simulation]] blocks
  max_clients = 100  # simulate various numbers of clients, up to this maximum
  repeat = 20  # simulate each (num_clients, strategy) combination this many times, to average out noise
  network_mu = 10.0  # network latency is modeled as max(0, N(mu, sigma))
  network_sigma = 2.0
  work_to_duration = 1.0
  control = "LockingServer"
  write_mu = 2.0  # write duration is modeled as max(0, N(mu, sigma))
  write_sigma = 1.0
  strategies = [  # one or more
    { type = "Constant", constant = 0.5 },
    { type = "FullJitteredExpo", base = 2.0, cap = 1000.0 },
    { type = "EqualJitteredExpo", base = 2.0, cap = 1000.0 },
  ]

  [[simulation]]
  title = "Read_Write_OCC_Example"
  max_clients = 100
  repeat = 30
  network_mu = 5.0
  network_sigma = 1.0
  write_mu = 0.0  # if write duration is negligible
  write_sigma = 0.0
  work_to_duration = 1.0
  control = "ReadWriteOCCServer"
  strategies = [
    { type = "Constant", constant = 0.0 },
    { type = "FullJitteredExpo", base = 5.0, cap = 2000.0 },
  ]
```

Then run:

```
backoff-simulator
```

You can name your config file something else and pass it via `--config-file path/to/file`.

For each `[[simulation]]` block, the simulator writes `<title>.png`, e.g.

![Metrics plot for LockingServer](/assets/images/Locking_Example_metrics.png)

showing work, duration and cost against number of clients, averaging across repetitions,
and `<title>_scatter.png`, e.g.

![Scatter plot for LockingServer](/assets/images/Locking_Example_scatter.png)

showing the distribution of client requests over time, for some simulation at `max_clients`.

It also writes representative event histories to stdout, e.g.

```text
Locking Example + Constant

  time    client_id  event_type             event_detail
------  -----------  ---------------------  --------------
  0               0  client_requests_write
  0               2  client_requests_write
  0               5  client_requests_write
  0               1  client_requests_write
  0               4  client_requests_write
  0               3  client_requests_write
  8               2  server_accepts
  8.17            4  server_rejects
  8.64            0  server_rejects
  9.55            2  server_commits
 12.32            5  server_accepts
 12.39            3  server_rejects
 13.85            1  server_rejects
 14.01            0  client_backs_off
 14.04            5  server_commits
 14.51            0  client_requests_write
 18.43            4  client_backs_off
 18.93            4  client_requests_write
 20.95            3  client_backs_off
 21.45            3  client_requests_write
 22.62            1  client_backs_off
 23.12            1  client_requests_write
 25.22            0  server_accepts
 26.72            4  server_rejects
 28.45            0  server_commits
 33.88            3  server_accepts
 35.6             1  server_rejects
 36.52            3  server_commits
 38.87            4  client_backs_off
 39.37            4  client_requests_write
 45.68            1  client_backs_off
 46.18            1  client_requests_write
 49.82            4  server_accepts
 51.52            4  server_commits
 57.53            1  server_accepts
 58.53            1  server_commits


Locking Example + EqualJitteredExpo

  time    client_id  event_type             event_detail
------  -----------  ---------------------  --------------
  0               0  client_requests_write
  0               2  client_requests_write
  0               5  client_requests_write
  0               1  client_requests_write
  0               4  client_requests_write
  0               3  client_requests_write
  6.82            2  server_accepts
  8.49            3  server_rejects
  8.6             1  server_rejects
 10.57            2  server_commits
 11.84            4  server_accepts
 12.2             0  server_rejects
 12.55            5  server_rejects
 13.37            4  server_commits
 17.39            1  client_backs_off
 19.24            1  client_requests_write
 21.91            3  client_backs_off
 22.58            0  client_backs_off
 22.75            5  client_backs_off
 22.96            3  client_requests_write
 24.02            0  client_requests_write
 24.23            5  client_requests_write
 28.14            1  server_accepts
 30.15            1  server_commits
 32.1             3  server_accepts
 34.18            3  server_commits
 34.34            0  server_accepts
 34.47            5  server_rejects
 36.23            0  server_commits
 46.4             5  client_backs_off
 48.99            5  client_requests_write
 59.03            5  server_accepts
 61.55            5  server_commits


Locking Example + FullJitteredExpo

  time    client_id  event_type             event_detail
------  -----------  ---------------------  --------------
  0               0  client_requests_write
  0               2  client_requests_write
  0               5  client_requests_write
  0               1  client_requests_write
  0               4  client_requests_write
  0               3  client_requests_write
  8.07            1  server_accepts
  8.23            2  server_rejects
  9.54            4  server_rejects
 10.04            0  server_rejects
 11.79            1  server_commits
 13.53            5  server_accepts
 13.57            5  server_commits
 13.59            3  server_accepts
 14.37            3  server_commits
 18.17            4  client_backs_off
 19.41            4  client_requests_write
 20.26            2  client_backs_off
 21.31            0  client_backs_off
 22.11            2  client_requests_write
 22.66            0  client_requests_write
 27.18            4  server_accepts
 31.76            4  server_commits
 34.23            0  server_accepts
 34.67            2  server_rejects
 36.48            0  server_commits
 46.65            2  client_backs_off
 48.49            2  client_requests_write
 55.36            2  server_accepts
 58.1             2  server_commits
```

## Concurrency controls

The server accepts or rejects requests, according to its concurrency control.
The app can simulate various controls.

The following controls are designed to manage contending writes.

**LockingServer**
- Parameters: `write_duration`, `write_sigma`.
- When it receives a request:
  - if available, it accepts it and becomes unavailable while writing.
  - if unavailable, it rejects it immediately.

**WriteOnlyOCCServer**
- Parameters: `write_duration`, `write_sigma`.
- The server stores a version number.
- When it receives a request, it notes the version number and tentatively writes.
- Then it checks the version again:
  - if different, it aborts
  - if the same, it commits and increments the version
- The writes are variable-duration.
  (Else a write would succeed just if no write was in progress when it arrived.
  So we'd end up with a locking server except it knowably does doomed-to-abort work.)

**ReadWriteOCCServer**
- Parameters: `write_duration`, `write_sigma`.
- The server stores a version number.
- A client first reads the version, then requests a write, passing the version.
- The server tentatively writes.
- Then it checks the version again:
  - if different, it aborts
  - if the same, it commits and increments the version

The following control is designed to manage overload.

**ThrottlingServer**
- Parameters: `window`, `limit`.
- The server accepts up to a limited number of requests in a sliding window.
- Any further requests are rejected immediately.

## Backoff strategies

The app can simulate various backoff strategies.

**Constant**
- e.g. for `constant = 3`, the client backs off for 3, 3, 3, ...
- this relies entirely on network and write variance to spread out requests

**Capped exponential backoff**
- `min(cap, base x 2^n)`
- e.g. for `cap = 10, b = 2` the client backs off for 2, 4, 8, 10, 10, ...

**Full jittered exponential backoff**
- `U(0, min(cap, base x 2^n))`
- i.e. a value picked uniformly at random between 0 and the capped exponential backoff

**Equal jittered exponential backoff**
- `min(cap, base x 2^n)) / 2 + U(0, min(cap, base x 2^n) / 2)`
- i.e. half the capped exponential backoff, plus a value picked uniformly at random between 0 and half the capped exponential backoff
