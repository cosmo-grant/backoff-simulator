checks:
  ty check --exclude inlined.py
  ruff check --fix --exclude inlined.py
  ruff format --exclude inlined.py
  marimo check --fix notebook.py inlined.py
  uv run backoff_simulator.py --repeat 2 --max-clients 3

inline:
  uv run python make_inlined.py

# same params as aws simulation
# aws sim doesn't model write delay, so we set it to 0
# aws repeats 100 times but my machines can't cope with that
aws-params:
  uv run backoff_simulator.py \
    --max-clients 200 \
    --constant 0 \
    --expo-base 5 \
    --expo-cap 2000 \
    --network-mu 10 \
    --network-sigma 2 \
    --write-mu 0 \
    --write-sigma 0 \
    --repeat 50
