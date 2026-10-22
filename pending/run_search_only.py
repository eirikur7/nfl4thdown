"""Re-run only step 2 of compute.ipynb (successive halving + final check), reusing everything step 1 saved in
results.json (heatmaps, sweeps, single strategies). Carries the top 5 strategy-7 cells and no-man's-land ranges."""
import copy, nbformat
from nbclient import NotebookClient
nb = nbformat.read("compute.ipynb", as_version=4)
state = nbformat.v4.new_code_cell('''res = M.load_results("results.json")
FG, best_x, best_n, NML_TOGO = res["fg_range"], res["best_x"], res["best_n"], 10
top_xy = [[int(x), int(y)] for x, y in res["heat"].stack().sort_values(ascending=False).index[:5]]
top_nml = [[35, 55], [35, 50], [40, 50], [30, 55], [40, 55]]   # top 5, but 40-55 (+0.57) instead of 25-55 (+0.61): team choice
res.update(top_xy=top_xy, top_nml=top_nml)
print("Step 1 reused from results.json. Top 5 (X, Y):", top_xy, " Top 5 no man's land:", top_nml)''')
RUN = [1, 3, None, 19, 21, 25]
cells = [state if i is None else copy.deepcopy(nb.cells[i]) for i in RUN]
tmp = nbformat.v4.new_notebook(cells=cells, metadata=nb.metadata)
NotebookClient(tmp, timeout=-1, kernel_name="python3", resources={"metadata": {"path": "."}}).execute()
for k, i in enumerate(RUN):
    if i is not None and nb.cells[i].cell_type == "code" and i >= 19:
        nb.cells[i].outputs = tmp.cells[k].outputs
nbformat.write(nb, "compute.ipynb")
print("done")
