import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Backoff Simulator
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
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

    So there’s a tradeoff.

    The _cost_ is an overall performance measure, got via an exchange rate: $\text{work-to-duration} \cdot \text{work} + \text{duration}$.

    Which backoff strategy minimizes the cost?

    There's a well-known aws blog post and simulation about this, focusing on Optimistic Concurrency Control:
    - https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter
    - https://github.com/aws-samples/aws-arch-backoff-simulator

    This app is based on those.
    But I re-implemented the simulation and added more controls.

    The blog post concludes:

    > The return on implementation complexity of using jittered backoff is huge,
    and it should be considered a standard approach for remote clients.

    The post has been influential.
    For example, the widely-used Python package [backoff](https://github.com/litl/backoff) defaults to full jitter, citing the post.
    But is it really the best strategy for _your_ use case?

    Let's explore.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simulation
    """)
    return


@app.cell(hide_code=True)
def _():
    import tomllib

    import marimo as mo

    from backoff_simulator.config import parse_spec
    from backoff_simulator.simulation import make_figures, make_tables, simulate

    return make_figures, make_tables, mo, parse_spec, simulate, tomllib


@app.cell(hide_code=True)
def _(mo):
    _default_toml = """\
    # Write toml to configure your simulations, then click "Submit".
    #
    # Each [[simulation]] block sets a concurrency control and one or more backoff strategies.
    #
    # The available controls are: ThrottlingServer, LockingServer, WriteOnlyOCCServer, ReadWriteOCCServer.
    # For ThrottlingServer, you must set: limit, window.
    # For the others, you must set: write_mu, write_sigma.
    #
    # The available backoff strategies are: Constant, FullJitteredExpo, EqualJitteredExpo.
    # For Constant, you must set: constant.
    # For the others, you must set: base, cap.
    #
    # Here's an example:

    [[simulation]]
    title = "LockingServer"  # anything you like, but must be unique across [[simulation]] blocks
    max_clients = 50  # simulate various numbers of clients, up to this maximum
    repeat = 20  # simulate each (num_clients, strategy) combination this many times, to average out noise
    network_mu = 10.0  # network latency is modeled as max(0, N(mu, sigma))
    network_sigma = 2.0
    work_to_duration = 1.0
    control = "LockingServer"
    write_mu = 2.0  # write duration is modeled as max(0, N(mu, sigma))
    write_sigma = 1.0
    strategies = [
      { type = "Constant", constant = 0.5 },
      { type = "FullJitteredExpo", base = 2.0, cap = 1000.0 },
      { type = "EqualJitteredExpo", base = 2.0, cap = 1000.0 },
    ]

    [[simulation]]
    title = "ReadWriteOCCServer"
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
    """

    editor = mo.ui.code_editor(value=_default_toml, language="toml", min_height=20).form()
    editor  # noqa: B018
    return (editor,)


@app.cell(hide_code=True)
def _(editor, make_figures, make_tables, mo, parse_spec, simulate, tomllib):
    mo.stop(editor.value is None)

    _raw = tomllib.loads(editor.value)
    _specs = [parse_spec(s) for s in _raw["simulation"]]

    results = []
    for _spec in _specs:
        _groups = simulate(_spec)
        _figs = make_figures(_groups, _spec)
        _tables = make_tables(_groups, _spec)
        results.append((_spec, _figs, _tables))
    return (results,)


@app.cell
def _(mo, results):
    mo.ui.tabs({spec.title: figs["metrics"] for spec, figs, _ in results})
    return


@app.cell
def _(mo, results):
    mo.ui.tabs({spec.title: figs["scatter"] for spec, figs, _ in results})
    return


@app.cell
def _(mo, results):
    def _concatenate(tables):
        output = ""
        for strategy, table in tables.items():
            output += f"**{strategy}**\n\n```\n{table}\n```\n\n"
        return mo.md(output)

    mo.ui.tabs({spec.title: _concatenate(tables) for spec, _, tables in results})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Concurrency controls

    The server accepts or rejects requests, according to its concurrency control.
    The app can simulate various controls.

    The following controls are designed to manage contending writes.

    **Locking**
    - When it receives a request:
      - if available, it accepts it and becomes unavailable for a while
      - if unavailable, it rejects it immediately.

    **Read-only optimistic concurrency control**
    - The server stores a version number.
    - When it receives a request, it notes the version number and tentatively writes.
    - Then it checks the version again:
      - if different, it aborts
      - if the same, it commits and increments the version
    - The writes are variable-duration.
      (Else a write would succeed just if no write was in progress when it arrived.
      So we'd end up with a locking server except it knowably does doomed-to-abort work.)

    **Read-write optimistic concurrency control**
    - The server stores a version number.
    - A client first reads the version, then requests a write, passing the version.
    - The server tentatively writes.
    - Then it checks the version again:
      - if different, it aborts
      - if the same, it commits and increments the version

    The following control is designed to manage overload.

    **Throttling**
    - The server accepts a limited number of requests in a sliding window.
    - Any further requests are rejected immediately.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Backoff strategies

    Each client backs off and retries until its request is accepted.
    The app can simulate various backoff strategies.
    ($n = 0, 1, 2, ...$ is the attempt number.)

    **Constant**
    - $c$
    - e.g. for $c=3$, the client backs off for $3, 3, 3, ...$
    - this relies entirely on network and write variance to spread out requests

    **Capped exponential backoff**
    - $min(c, b \cdot 2^n)$
    - e.g. for $c=10, b=2$ the client backs off for $2, 4, 8, 10, 10, ...$

    **Full jittered exponential backoff**
    - $U(0, min(c, b \cdot 2^n))$
    - i.e. a value picked uniformly at random between 0 and the capped exponential backoff

    **Equal jittered exponential backoff**
    - $min(c, b \cdot 2^n)) / 2 + U(0, min(c, b \cdot 2^n) / 2)$
    - i.e. half the capped exponential backoff, plus a value picked uniformly
      at random between 0 and half the capped exponential backoff
    """)
    return


if __name__ == "__main__":
    app.run()
