checks:
  ty check
  ruff check --fix
  ruff format

# same params as aws simulation
# aws doesn't model write delay, so we set it to 0
# aws repeats 100 times but my machine can't cope with that
aws-params:
  uv run backoff-simulator --config-file aws_simulation.toml
