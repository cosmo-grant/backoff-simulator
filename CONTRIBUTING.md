# CONTRIBUTING

## Division of labour

- simulation code goes in `backoff_simulator.py`
- marimo-specific stuff (ui elements, markdown etc) goes in `notebook.py`
- `notebook.py` imports what it needs from `backoff_simulator.py`
- run `just inline` to generate a version of the notebook with imports in-lined
- share `inlined.py` on molab

## Notes

- which python version you get on molab seems to vary, and not be configurable
