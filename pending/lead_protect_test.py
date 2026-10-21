"""Quick test: on top of the best strategy, never go for it when leading with <= N drives left at pos >= Z."""
import sys, json, numpy as np, model as M
from multiprocessing import get_context
m = M.estimate(M.load_data()); r = M.load_results("results.json"); fg = r["fg_range"]; best = r["best"]
opp = sys.argv[1] if len(sys.argv) > 1 else "equal"
N_GAMES, SEED = 50_000, 4000
def protect_fp(pol, N, Z):
    def p(m_, togo, pos, diff, left, rng):
        a = pol(m_, togo, pos, diff, left, rng)
        if a == "go" and diff > 0 and left <= N and pos >= Z:
            return M.kick(pos, fg)
        return a
    return p
base_pol = M.build(best, fg)
variants = {"best": base_pol}
for N in (1, 2, 3):
    for Z in (50, 60, 70, 80):
        variants[f"N={N}, pos>={Z}"] = protect_fp(base_pol, N, Z)
names = list(variants)
def job(k): return M.run(m, variants[names[k]], opp, n=N_GAMES, seed=SEED)
with get_context("fork").Pool(2) as pool:
    outs = dict(zip(names, [o for o, _ in pool.map(job, range(len(names)))]))
coach = M.run(m, M.coach, opp, n=N_GAMES, seed=SEED)[0]
print("best vs coaches: %.2f [%.2f, %.2f]" % M.ci(100 * (outs["best"] - coach)))
for k in names[1:]:
    print(f"{k:14s} minus best: %.3f [%.3f, %.3f]" % M.ci(100 * (outs[k] - outs["best"])))
