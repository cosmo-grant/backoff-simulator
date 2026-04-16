import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


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
    mo.stop(form.value is None, mo.md("**Set parameters and click submit.**"))
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


if __name__ == "__main__":
    app.run()
