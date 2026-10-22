"""Extension: the best strategy against a weak and a strong opponent, found exactly like the equal-opponent one.

    python opponents_search.py        # about 1 hour; updates results.json (opponent curves) and saves results_opponents.json

1. Opponent curves (compute.ipynb step 3) with the current model: gain of "go if <= X" over the coaches, per opponent.
2. For weak and strong: strategy-7 heatmap (100,000 paired games per cell) and dead-zone heatmap (10,000 per range)
   against that opponent, the top 5 of each into successive halving (1,000 -> 64,000 games), then the final check on
   50,000 fresh paired games. The equal opponent is the main analysis (results.json)."""
import numpy as np, pandas as pd
from multiprocessing import get_context
import model as M

m = M.estimate(M.load_data())
res = M.load_results("results.json")
FG = res["fg_range"]
OPPS = ["weak", "equal", "strong"]
G = {}


def _run(args):
    key, o, n, seed = args
    return M.run(m, G[key], o, n=n, seed=seed)[0].astype(np.float32)


def parallel(jobs):
    with get_context("fork").Pool(2) as pool:
        return pool.map(_run, jobs)


# ---- 1. opponent curves (as compute.ipynb step 3)
X_OPP, N_OPP = [0, 1, 2, 3, 4, 6, 8, 10, 15], 40_000
rows = [str(x) for x in X_OPP] + ["always"]
G.update({"coach": M.coach, "always": M.always}, **{str(X): M.strategy([M.short_yardage(X)], FG) for X in X_OPP})
aggr, aggr_lo, aggr_hi = (pd.DataFrame(index=rows, columns=OPPS, dtype=float) for _ in range(3))
aggr_base = {}
for k, o in enumerate(OPPS):
    outs = parallel([(key, o, N_OPP, 1000 + k) for key in ["coach"] + rows])
    base, outs = outs[0].astype(float), outs[1:]
    aggr_base[o] = base.mean()
    for key, out in zip(rows, outs):
        aggr.loc[key, o], aggr_lo.loc[key, o], aggr_hi.loc[key, o] = M.ci(100 * (out - base))
best_x_opp = {o: aggr[o].drop("always").idxmax() for o in OPPS}
G["single"] = M.strategy([M.short_yardage(res["best_x"])], FG)
G["best"] = M.build(res["best"], FG)
opp_res = pd.DataFrame({o: M.compare(m, {f"best single (≤{res['best_x']} yd)": G["single"], "best combination": G["best"]},
                                     o, n=30_000, seed=1100 + k)["diff (pp)"] for k, o in enumerate(OPPS)})
res.update(aggr=aggr, aggr_lo=aggr_lo, aggr_hi=aggr_hi, aggr_base=aggr_base, best_x_opp=best_x_opp, opp_res=opp_res)
M.save_results(res, "results.json")
print("Opponent curves:", {o: round(aggr_base[o], 3) for o in OPPS}, best_x_opp, flush=True)

# ---- 2. weak and strong: heatmaps, top 5, successive halving, final check
out = {"equal": {"best": res["best"], "best_name": res["best_name"], "final": res["final"],
                 "top_xy": res["top_xy"], "top_nml": res["top_nml"], "coaches": aggr_base["equal"]}}
A_VALS, B_VALS = [25, 30, 35, 40, 45, 50], [35, 40, 45, 50, 55, 60, 65]
for k, o in enumerate(["weak", "strong"]):
    seed0 = 1700 + 1000 * k
    heat, tied = M.strategy7_grid(m, FG, list(range(1, 9)), list(range(10, 101, 10)), n=100_000, seed=12, opp=o)
    top_xy = [[int(x), int(y)] for x, y in heat.stack().sort_values(ascending=False).index[:5]]
    best_x = int(heat[100].idxmax())
    G["nml_base"] = M.strategy([M.short_yardage(best_x)], FG)
    cells = [(a, b) for a in A_VALS for b in B_VALS if a < b]
    for a, b in cells:
        G[f"nml{a}-{b}"] = M.strategy([M.short_yardage(best_x), M.no_mans_land(a, b, 10)], FG)
    outs = parallel([("nml_base", o, 10_000, 14)] + [(f"nml{a}-{b}", o, 10_000, 14) for a, b in cells])
    nml_base = outs[0].mean()
    nml = pd.DataFrame(index=A_VALS, columns=B_VALS, dtype=float)
    for (a, b), o_ in zip(cells, outs[1:]):
        nml.loc[a, b] = o_.mean()
    top_nml = [[int(a), int(b)] for a, b in nml.stack().sort_values(ascending=False).index[:5]]
    print(o, "top strategy 7:", top_xy, "top dead zone:", top_nml, flush=True)

    cands = M.candidates(top_xy, top_nml)
    history, games, last = M.successive_halving(m, cands, FG, n=1_000, seed=seed0, opp=o)
    final_ids = sorted(history[-1], key=history[-1].get, reverse=True)[:5]
    single_name = f"Best single rule: 5 (X = {best_x})"
    pols = {"1. Real coaches": M.coach, single_name: M.strategy([M.short_yardage(best_x)], FG)}
    for i in final_ids:
        pols[M.label(cands[i])] = M.build(cands[i], FG)
    pols.setdefault("Equal-opponent winner: " + res["best_name"], M.build(res["best"], FG))
    final = M.compare(m, pols, o, n=50_000, seed=seed0 + 200)
    combos = final.drop(["1. Real coaches", single_name])
    combos = combos[~combos.index.str.startswith("Equal-opponent winner")]
    best_name = combos["diff (pp)"].idxmax()
    best = cands[next(i for i in final_ids if M.label(cands[i]) == best_name)]
    out[o] = {"best": best, "best_name": best_name, "final": final, "top_xy": top_xy, "top_nml": top_nml,
              "heat": heat, "heat_tied": tied, "nml": nml, "nml_base": nml_base, "best_x": best_x,
              "coaches": aggr_base[o], "handbook": M.handbook(m, M.build(best, FG))}
    print(final[["p̂", "diff (pp)", "diff lower", "diff upper"]].round(3).to_string(), flush=True)
    print(o, "best:", best_name, flush=True)

flat = {f"{o}/{k}": v for o, d in out.items() for k, v in d.items()}
M.save_results(flat, "results_opponents.json")
print("Saved results_opponents.json")
