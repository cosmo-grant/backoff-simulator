checks:
  ty check
  ruff check --fix
  ruff format
  uv run backoff_simulator.py --repeat 2 --max-clients 3
