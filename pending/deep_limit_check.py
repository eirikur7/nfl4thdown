"""Check: 'go if <= 6' vs 'go if <= 6, but inside own Z only if <= 3', on many paired games.
Also counts how often the two strategies actually make a different call."""
import random, numpy as np, model as M
from multiprocessing import get_context
m = M.estimate(M.load_data()); FG = M.load_results("results.json")["fg_range"]
N, SEEDS = 200_000, [5100, 5200]
def rule(X, d, Z): return lambda togo, pos, diff, left: togo <= X and (pos <= Z or togo <= d)
def job(args):
    Z, seed = args
    a = M.run(m, M.strategy([M.short_yardage(6)], FG), n=N, seed=seed)[0]
    b = M.run(m, M.strategy([rule(6, 3, Z)], FG), n=N, seed=seed)[0]
    return a - b
if __name__ == "__main__":
    # how often: 4th & 4-6 inside own 10 / 20 / 30 per game (our team, 'go if <= 6')
    cnt = {90: 0, 80: 0, 70: 0}; G = 20_000
    def counting(m_, togo, pos, diff, left, rng):
        for Z in cnt:
            if 4 <= togo <= 6 and pos > Z: cnt[Z] += 1
        return "go" if togo <= 6 else M.kick(pos, FG)
    for i in range(G):
        M.game(m, counting, "equal", random.Random(10_000 + i))
    for Z in cnt: print(f"4th & 4-6 inside own {100 - Z}: {cnt[Z] / G:.3f} times per game")
    with get_context("fork").Pool(2) as pool:
        for Z in (90, 80, 70):
            d = np.concatenate(pool.map(job, [(Z, s) for s in SEEDS]))
            print(f"inside own {100 - Z}: (go if <= 6 anywhere) - (only <= 3 there) = %.3f pp [%.3f, %.3f]  (n = {len(d):,})" % M.ci(100 * d))
