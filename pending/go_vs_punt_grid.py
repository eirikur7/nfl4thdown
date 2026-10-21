"""Run: python deep_tradeoff.py  (about 30 s, saves deep_tradeoff.csv, shown in results.ipynb)

Why (not) go for it deep in our own half? For 4th & d at our own spot, compare punting with going for it:
the opponent's next drive after a punt vs after a failed attempt, and the net points of
(rest of our drive + opponent's next drive) for each choice."""
import sys, random, numpy as np, pandas as pd

import model as M
m = M.estimate(M.load_data()); FG = M.load_results("results.json")["fg_range"]
Y = m["yards"]["equal"]
pol = M.strategy([M.short_yardage(4)], FG)          # what we do on later 4th downs of this drive
N = 10_000

def opp_drive(start, rng):
    pts_them, pts_us, nxt, end = M.drive(m, start, M.coach, Y, 0, 6, rng)
    return pts_them, pts_us                               # points for opponent, points for us (safety)

rows = []
for own in [10, 20, 30, 40, 50]:
    pos = 100 - own
    for d in [1, 2, 3, 4, 5, 6, 8]:
        if d > pos: continue
        rng = random.Random(own * 100 + d)
        # --- punt
        p_td, p_sc, p_net = 0, 0, []
        for _ in range(N):
            start = M.punt(m, pos, rng)
            them, us = opp_drive(start, rng)
            p_td += them >= 6; p_sc += them > 0; p_net.append(us - them)
        # --- go for it
        conv, f_td, f_sc, fails, g_net = 0, 0, 0, 0, []
        ys, p_to = Y[(3, min(d, 11))]
        for _ in range(N):
            if rng.random() < p_to:                        # turnover on the attempt
                start, ok = max(1, 100 - pos), False
            else:
                y = ys[int(rng.random() * len(ys))]
                newpos = pos - y
                if newpos <= 0:                             # touchdown on the play
                    g_net.append(6 + (rng.random() < m["p_xp"])); conv += 1; continue
                if newpos >= 100:                           # safety
                    g_net.append(-2); continue
                ok = y >= d
                start = 100 - newpos
            if ok:
                conv += 1
                a, b, opp_start, _ = M.drive(m, newpos, pol, Y, 0, 6, rng)   # rest of our drive
                them, us = opp_drive(opp_start, rng)
                g_net.append(a - b + us - them)
            else:
                fails += 1
                them, us = opp_drive(start, rng)
                f_td += them >= 6; f_sc += them > 0; g_net.append(us - them)
        rows.append({"our own": own, "4th &": d, "P(convert)": conv / N,
                     "opp TD after failed go": f_td / max(fails, 1), "opp TD after punt": p_td / N,
                     "opp score after failed go": f_sc / max(fails, 1), "opp score after punt": p_sc / N,
                     "net pts: go": np.mean(g_net), "net pts: punt": np.mean(p_net)})
t = pd.DataFrame(rows)
t["go − punt (pts)"] = t["net pts: go"] - t["net pts: punt"]
pd.set_option("display.width", 250)
print(t.round(3).to_string(index=False))
t.to_csv(sys.argv[1], index=False)
