import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Backoff Simulator

    Simulate backoff strategies for contending writes over the network.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from backoff_simulator import make_figures, make_tables, simulate

    return make_figures, make_tables, simulate


@app.cell
def _(mo):
    form = (
        mo.Html(
            """
        <style>
            .params-table td:nth-child(2) input {{ width: 80px; }}
            .params-table td {{ padding: 4px 8px; }}
        </style>
        <table class="params-table">
            <tr><td><strong>Max clients</strong></td><td>{max_clients}</td></tr>
            <tr><td><strong>Expo base</strong></td><td>{expo_base}</td></tr>
            <tr><td><strong>Expo cap</strong></td><td>{expo_cap}</td></tr>
            <tr><td><strong>Network mu</strong></td><td>{network_mu}</td></tr>
            <tr><td><strong>Network sigma</strong></td><td>{network_sigma}</td></tr>
            <tr><td><strong>Write mu</strong></td><td>{write_mu}</td></tr>
            <tr><td><strong>Write sigma</strong></td><td>{write_sigma}</td></tr>
            <tr><td><strong>Repeat count</strong></td><td>{repeat}</td></tr>
            <tr><td><strong>Requests over duration</strong></td><td>{requests_over_duration}</td></tr>
        </table>
        """
        )
        .batch(
            max_clients=mo.ui.number(start=1, stop=100, step=1, value=50),
            expo_base=mo.ui.number(start=1, stop=10, step=1, value=2),
            expo_cap=mo.ui.number(start=2, stop=10, step=1, value=10),
            network_mu=mo.ui.number(start=2, stop=10, step=1, value=10),
            network_sigma=mo.ui.number(start=1, stop=5, step=1, value=2),
            write_mu=mo.ui.number(start=2, stop=10, step=1, value=5),
            write_sigma=mo.ui.number(start=1, stop=5, step=1, value=1),
            repeat=mo.ui.number(start=1, stop=100, step=1, value=20),
            requests_over_duration=mo.ui.number(start=0.1, stop=10, step=0.1, value=1),
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
    figs = make_figures(groups, params["max_clients"], params["requests_over_duration"])
    tables = make_tables(groups, params["max_clients"])
    return figs, tables


@app.cell
def _(figs, mo):
    metrics_tabs = mo.ui.tabs({control: mo.as_html(fig) for control, fig in figs["metrics"]})
    metrics_tabs
    return


@app.cell
def _(figs, mo):
    scatter_tabs = mo.ui.tabs({control: mo.as_html(fig) for control, fig in figs["scatter"]})
    scatter_tabs
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
    ## Background
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Set up

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
    It's set by a work-to-duration exchange rate:
    - say, 1 if you think 1 extra request and 1 extra ms are equally bad
    - or 5 if you think 5 extra requests and 1 extra ms are equally bad
    - or 0.2 if you think 1 extra request and 5 extra ms are equally bad.

    There's a famous aws blog post and simulation script about this:
    - https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter
    - https://github.com/aws-samples/aws-arch-backoff-simulator

    This app is based on those.
    But I re-implemented the simulation (til about heap-based priority queues) and added a few bells and whistles.

    Explore for yourself!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Contention models

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
    - The writes are variable-duration. (Else a write would succeed just if no write was in progress when it arrived. So we'd end up with a locking server except it knowably does doomed-to-abort work.)

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
    ### Backoff strategies

    How do clients back off?
    I focus on the following strategies, where $n$ is the attempt number.

    **Capped exponential backoff:**
    - $min(c, b \cdot 2^n)$
    - e.g. for $c=10, b=2$ this gives $2, 4, 8, 10, 10, ...$

    **Full jittered exponential backoff:**
    - $U(0, min(c, b \times 2^n))$
    - i.e. a value picked uniformly at random between 0 and the capped exponential backoff
    """)
    return


if __name__ == "__main__":
    app.run()
