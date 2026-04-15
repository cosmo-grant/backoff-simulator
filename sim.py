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
    form = mo.ui.dictionary(
        {
            "max_clients": mo.ui.number(start=1, stop=100, step=1, value=50, label="max clients"),
            "expo_base": mo.ui.number(start=1, stop=10, step=1, value=2, label="expo base"),
            "expo_cap": mo.ui.number(start=2, stop=10, step=1, value=10, label="expo cap"),
            "network_mu": mo.ui.number(start=2, stop=10, step=1, value=10, label="network mu"),
            "network_sigma": mo.ui.number(start=1, stop=5, step=1, value=2, label="network sigma"),
            "write_mu": mo.ui.number(start=2, stop=10, step=1, value=5, label="write mu"),
            "write_sigma": mo.ui.number(start=1, stop=5, step=1, value=1, label="write sigma"),
            "repeat": mo.ui.number(start=1, stop=100, step=1, value=20, label="repeat count"),
            "requests_over_duration": mo.ui.number(start=0.1, stop=10, step=0.1, value=1, label="requests over duration"),
        }
    ).form()

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
