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
            "max_clients": mo.ui.slider(1, 100, value=10, label="max clients"),
            "expo_base": mo.ui.slider(1, 10, value=2, label="expo base"),
            "expo_cap": mo.ui.slider(2, 10, value=10, label="expo cap"),
            "network_mu": mo.ui.slider(2, 10, value=10, label="network mu"),
            "network_sigma": mo.ui.slider(1, 5, value=2, label="network sigma"),
            "write_mu": mo.ui.slider(2, 10, value=5, label="write mu"),
            "write_sigma": mo.ui.slider(1, 5, value=1, label="write sigma"),
            "repeat": mo.ui.slider(1, 100, value=10, label="how many times to run each configuration"),
            "requests_over_duration": mo.ui.slider(0.1, 10, value=1, label="exchange rate"),
        }
    ).form()

    form  # noqa: B018
    return (form,)


@app.cell
def _(form, mo, simulate, make_figures, make_tables):
    mo.stop(form.value is None, mo.md("**Set parameters and click submit.**"))
    params = form.value
    groups = simulate(**params)
    figs = make_figures(groups, params["max_clients"], params["requests_over_duration"])
    tables = make_tables(groups, params["max_clients"])
    mo.vstack(
        [
            *[mo.as_html(fig) for fig in figs],
            mo.md(f"```\n{tables}\n```"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
