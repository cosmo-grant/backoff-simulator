from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt

from .config import load_config
from .simulation import make_figures, make_tables, simulate

parser = ArgumentParser(description="Run backoff simulations configured via a toml file.")
parser.add_argument("--config-file", type=Path, default="simulations.toml", help="Path to config file.")


def app() -> None:
    args = parser.parse_args()

    specs = load_config(args.config_file)
    for spec in specs:
        groups = simulate(spec)

        figures = make_figures(groups, spec)
        for kind, fig in figures.items():
            filename = f"{spec.control}_{kind}.png"
            fig.savefig(filename)
            plt.close(fig)

        tables = make_tables(groups, spec)
        for strategy, table in tables.items():
            print(f"\n{spec.control} + {strategy}\n")
            print(table)
            print()


if __name__ == "__main__":
    app()
