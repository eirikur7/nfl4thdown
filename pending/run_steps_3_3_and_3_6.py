"""Re-run only compute.ipynb steps 3.3 (strategy-7 heatmap) and 3.6 (single strategies) with the current model,
keeping everything else in results.json; writes those cells' outputs back into compute.ipynb."""
import copy, nbformat
from nbclient import NotebookClient
nb = nbformat.read("compute.ipynb", as_version=4)
state = nbformat.v4.new_code_cell('''res = M.load_results("results.json")
FG, best_x, best_n, top_nml, NML_TOGO = res["fg_range"], res["best_x"], res["best_n"], res["top_nml"], 10''')
run_ids = [1, 3, None, 11, 17, 25]
cells = [state if i is None else copy.deepcopy(nb.cells[i]) for i in run_ids]
tmp = nbformat.v4.new_notebook(cells=cells, metadata=nb.metadata)
NotebookClient(tmp, timeout=-1, kernel_name="python3", resources={"metadata": {"path": "."}}).execute()
for k, i in enumerate(run_ids):
    if i in (11, 17, 25):
        nb.cells[i].outputs = tmp.cells[k].outputs
nbformat.write(nb, "compute.ipynb")
print("done")
