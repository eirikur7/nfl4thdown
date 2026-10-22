"""Run compute.ipynb top to bottom EXCEPT step 3 (opponent strength, cell 23). The opponent results already in
results.json are kept. Outputs are written back into compute.ipynb (cell 23 keeps its old output)."""
import copy, nbformat
from nbclient import NotebookClient
nb = nbformat.read("compute.ipynb", as_version=4)
OPP_CELL, SAVE_CELL = 23, 25
keep = nbformat.v4.new_code_cell('''_old = M.load_results("results.json")
for k in ["aggr", "aggr_lo", "aggr_hi", "aggr_base", "best_x_opp", "opp_res"]:
    if k in _old: res[k] = _old[k]
print("Opponent step skipped; kept the previous opponent results")''')
HEAT_CELL = 11
dump_heat = nbformat.v4.new_code_cell('''import plots as P, matplotlib.pyplot as plt
P.plt.show = lambda: plt.gcf().savefig("pending/heat_strategy7_latest.png", dpi=120, bbox_inches="tight")
P.short_and_close({"heat": heat, "heat_tied": heat_tied, "top_xy": top_xy}); plt.close("all")
heat.to_csv("pending/heat_strategy7_latest.csv")
print("heatmap saved")''')
ids = [i for i in range(len(nb.cells)) if i != OPP_CELL]
cells = []
for i in ids:
    if i == SAVE_CELL:
        cells.append(keep)
    cells.append(copy.deepcopy(nb.cells[i]))
    if i == HEAT_CELL:
        cells.append(dump_heat)
tmp = nbformat.v4.new_notebook(cells=cells, metadata=nb.metadata)
NotebookClient(tmp, timeout=-1, kernel_name="python3", resources={"metadata": {"path": "."}}).execute()
src = [c for c in tmp.cells if c is not keep and c is not dump_heat]
for i, c in zip(ids, src):
    nb.cells[i].outputs = c.outputs
    if "execution_count" in c: nb.cells[i].execution_count = c.execution_count
nbformat.write(nb, "compute.ipynb")
print("done")
