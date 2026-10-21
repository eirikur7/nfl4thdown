"""Re-run only the strategy-7 heatmap (compute.ipynb step 3.3) with the current model, plot it."""
import random, json, collections, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import get_context
import model as M, plots as P
m = M.estimate(M.load_data()); r = M.load_results("results.json"); FG = r["fg_range"]
rng = random.Random(1); c = collections.Counter()
for _ in range(200_000):
    s = M.punt(m, 75, rng); c["TD" if s == 0 else "muffTD" if s < 0 else "muff" if s > 100 else "normal"] += 1
print("punts from own 25:", {k: v / 2000 for k, v in c.items()}, "%")
X_VALS, Y_VALS = [1, 2, 3, 4, 5, 6], list(range(20, 101, 5))
cells = [(X, Y) for X in X_VALS for Y in Y_VALS]
def job(xy): return M.run(m, M.strategy([M.short_and_close(*xy)], FG), n=20_000, seed=12)[0].mean()
with get_context("fork").Pool(2) as pool:
    vals = pool.map(job, cells)
heat = pd.DataFrame(index=X_VALS, columns=Y_VALS, dtype=float)
for (X, Y), v in zip(cells, vals): heat.loc[X, Y] = v
top_xy = [[int(x), int(y)] for x, y in heat.stack().sort_values(ascending=False).index[:10]]
print("Top 10:", top_xy); print(heat.round(3).to_string())
heat.to_csv("pending/heat_strategy7_with_punt_returns.csv")
P.plt.show = lambda: plt.gcf().savefig("pending/heat_strategy7_with_punt_returns.png", dpi=110, bbox_inches="tight")
P.short_and_close({"heat": heat, "top_xy": top_xy})
