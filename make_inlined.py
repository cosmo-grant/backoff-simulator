"""
Generate app.py by inlining backoff_simulator.py into notebook.py.
This lets me share the notebook on molab.
"""

import re
from pathlib import Path
from textwrap import dedent, indent

ROOT = Path(__file__).parent
NOTEBOOK = ROOT / "notebook.py"
MODULE = ROOT / "backoff_simulator.py"
INLINED = ROOT / "inlined.py"

IMPORT_CELL = dedent("""\
    @app.cell
    def _():
        from backoff_simulator import make_figures, make_tables, simulate

        return make_figures, make_tables, simulate
    """)


def strip_main_block(source: str) -> str:
    return re.sub(r'\nif __name__ == "__main__":.*', "", source, flags=re.DOTALL)


def make_inlined_cell(module_source: str) -> str:
    """Build a replacement cell with the module source inlined."""
    indented = indent(module_source, "    ")
    return f"@app.cell(hide_code=True)\ndef _():\n{indented}\n    return make_figures, make_tables, simulate\n"


def main() -> None:
    notebook = NOTEBOOK.read_text()
    module = MODULE.read_text()
    module = strip_main_block(module)
    module = module.strip() + "\n"
    assert IMPORT_CELL in notebook, f"Could not find the import cell in {NOTEBOOK}. Has it changed?"
    inlined = notebook.replace(IMPORT_CELL, make_inlined_cell(module))
    INLINED.write_text(inlined)
    print(f"Wrote {INLINED}")


if __name__ == "__main__":
    main()
