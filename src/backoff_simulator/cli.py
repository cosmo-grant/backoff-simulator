from argparse import ArgumentParser

from .simulation import run

parser = ArgumentParser()
parser.add_argument("--max-clients", type=int, default=50)
parser.add_argument("--constant", type=float, default=0.5)
parser.add_argument("--expo-base", type=float, default=2)
parser.add_argument("--expo-cap", type=float, default=1000)
parser.add_argument("--network-mu", type=float, default=10)
parser.add_argument("--network-sigma", type=float, default=2)
parser.add_argument("--write-mu", type=float, default=2)
parser.add_argument("--write-sigma", type=float, default=1)
parser.add_argument("--repeat", type=int, default=20)
parser.add_argument("--work-to-duration", type=float, default=1)


def app() -> None:
    args = parser.parse_args()
    args = vars(args)

    # crude validation, e.g. max_clients should not be 0, but good enough
    if not all(v >= 0 for v in args.values()):
        parser.error(f"All values must be non-negative.\nGiven: {args}")

    run(**args)


if __name__ == "__main__":
    app()
