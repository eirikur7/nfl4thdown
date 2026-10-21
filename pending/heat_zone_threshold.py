"""New strategy 7 (proposal): go if <= X yards to go, but deep in our own end (pos > Z) only if <= d.
Grid over X, d and Z; 20,000 games per cell (seed 12, as the strategy-7 heatmap). Saves csv + png in pending/."""
import itertools, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import get_context
import model as M, plots as P
m = M.estimate(M.load_data()); FG = M.load_results("results.json")["fg_range"]
X_VALS, D_VALS, Z_VALS = [3, 4, 5, 6, 8], [0, 1, 2, 3, 4, 5], [90, 80, 70, 60]   # Z 90 = inside own 10, ..., 60 = inside own 40
def rule(X, d, Z):
    return lambda togo, pos, diff, left: togo <= X and (pos <= Z or togo <= d)
cells = [(X, d, Z) for X, d, Z in itertools.product(X_VALS, D_VALS, Z_VALS) if d < X]
def job(c): return M.run(m, M.strategy([rule(*c)], FG), n=20_000, seed=12)[0].mean()
if __name__ == "__main__":
    with get_context("fork").Pool(2) as pool:
        vals = pool.map(job, cells)
    t = pd.DataFrame([(*c, v) for c, v in zip(cells, vals)], columns=["X", "d", "Z", "p"])
    nolim = {X: M.run(m, M.strategy([M.short_yardage(X)], FG), n=20_000, seed=12)[0].mean() for X in X_VALS}
    t.to_csv("pending/heat_zone_threshold.csv", index=False)
    top = t.sort_values("p", ascending=False).head(10)
    print(top.to_string(index=False)); print("no limit:", {k: round(v, 4) for k, v in nolim.items()})
    fig, axs = plt.subplots(1, 4, figsize=(20, 4.4), sharey=True)
    for ax, Z in zip(axs, Z_VALS):
        h = t[t.Z == Z].pivot(index="X", columns="d", values="p").reindex(index=X_VALS, columns=D_VALS)
        h["no limit"] = [nolim[X] for X in X_VALS]
        best = [[int(r.X), int(r.d)] for r in top[top.Z == Z].itertuples()]
        im = P.heatmap(ax, h, lambda v: f"{v:.3f}", 0.50, 0.56, best)
        ax.set(title=f"Inside own {100 - Z}: go only if ≤ d", xlabel="d (yards to go allowed deep)")
    axs[0].set_ylabel("X: max yards to go elsewhere")
    fig.colorbar(im, ax=axs, label="Win probability", fraction=0.015)
    fig.suptitle("New strategy 7: go if ≤ X yards, but deep in our own end only if ≤ d (boxed: overall top 10)", y=1.03)
    fig.savefig("pending/heat_zone_threshold.png", dpi=110, bbox_inches="tight")
