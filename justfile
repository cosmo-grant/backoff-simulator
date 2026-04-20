checks:
  ty check --exclude notebook.py
  ruff check --fix
  ruff format
  marimo check --fix notebook.py
  uv run backoff-simulator --repeat 2 --max-clients 3

# same params as aws simulation
# aws doesn't model write delay, so we set it to 0
# aws repeats 100 times but my machine can't cope with that
aws-params:
  uv run backoff-simulator \
    --max-clients 200 \
    --constant 0 \
    --expo-base 5 \
    --expo-cap 2000 \
    --network-mu 10 \
    --network-sigma 2 \
    --write-mu 0 \
    --write-sigma 0 \
    --repeat 50
