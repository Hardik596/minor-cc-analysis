"""One-off: execute the notebook in place via nbclient. Delete after running.

nbconvert --execute drags in the pandoc templating chain, which is not needed to
run cells and fails without pandocfilters. nbclient runs the kernel directly.
"""
import sys

import nbformat
from nbclient import NotebookClient

PATH = "cc_analysis.ipynb"

nb = nbformat.read(PATH, as_version=4)
client = NotebookClient(nb, timeout=3600, kernel_name="python3",
                        resources={"metadata": {"path": "."}})
try:
    client.execute()
finally:
    nbformat.write(nb, PATH)

errors = [o for cell in nb.cells if cell.cell_type == "code"
          for o in cell.get("outputs", []) if o.get("output_type") == "error"]
figures = sum(1 for cell in nb.cells if cell.cell_type == "code"
              for o in cell.get("outputs", [])
              if "image/png" in (o.get("data") or {}))
print(f"executed; figures embedded: {figures}; errors: {len(errors)}")
for e in errors[:3]:
    print(" ", e.get("ename"), e.get("evalue"))
sys.exit(1 if errors else 0)
