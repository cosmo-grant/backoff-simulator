checks:
  ty check
  ruff check --fix
  ruff format

aws-params:
  uv run backoff-simulator --config-file aws_simulation.toml
