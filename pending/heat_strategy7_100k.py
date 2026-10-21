"""Strategy 7 heatmap: go if <= X yards to go and pos <= Y, else kick. X = 1-8, Y = 10-100 (10-yd steps).
100,000 paired games per cell (same seed for every cell). Boxed: cells NOT significantly worse than the best
cell (95% CI of the paired difference reaches 0)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import get_context
import model as M, plots as P
m = M.estimate(M.load_data()); FG = M.load_results("results.json")["fg_range"]
X_VALS, Y_VALS, N, SEED = list(range(1, 9)), list(range(10, 101, 10)), 100_000, 12
cells = [(X, Y) for X in X_VALS for Y in Y_VALS]
def job(c): return M.run(m, M.strategy([M.short_and_close(*c)], FG), n=N, seed=SEED)[0].astype(np.float32)
if __name__ == "__main__":
    with get_context("fork").Pool(2) as pool:
        outs = dict(zip(cells, pool.map(job, cells)))
    heat = pd.DataFrame(index=X_VALS, columns=Y_VALS, dtype=float)
    for (X, Y), o in outs.items(): heat.loc[X, Y] = o.mean()
    best = max(outs, key=lambda c: outs[c].mean())
    tied, rows = [], []
    for c, o in outs.items():
        d, lo, hi = M.ci(100 * (o.astype(float) - outs[best]))
        rows.append((*c, o.mean(), d, lo, hi))
        if hi >= 0: tied.append(list(c))
    t = pd.DataFrame(rows, columns=["X", "Y", "p", "vs best (pp)", "lower", "upper"])
    t.to_csv("pending/heat_strategy7_100k.csv", index=False)
    print("best cell:", best, round(outs[best].mean(), 4)); print("tied with best:", tied)
    print(heat.round(4).to_string())
    print("SE of one cell's win prob ≈ %.4f" % (outs[best].std() / np.sqrt(len(outs[best]))))
    fig, ax = plt.subplots(figsize=(11, 5.2))
    im = P.heatmap(ax, heat, lambda v: f"{v:.3f}", 0.44, 0.56, tied)
    ax.set(xlabel="Y: go only if pos ≤ Y (100 = anywhere)", ylabel="X: max yards to go",
           title=f"Strategy 7, {N:,} paired games per cell (boxed: not significantly worse than the best cell)")
    fig.colorbar(im, label="Win probability")
    fig.savefig("pending/heat_strategy7_100k.png", dpi=120, bbox_inches="tight")
