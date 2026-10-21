"""Best single strategy and best combination per opponent strength (weak / equal / strong).

Extension of compute.ipynb. Uses the same candidates, field-goal range and rule parameters as the
main (equal-opponent) run, read from results.json, so the three opponents are compared like for like.

    python opponents_compute.py weak      # ~20 min (combination search), saves results_opp_weak.json
    python opponents_compute.py strong
    python opponents_compute.py equal     # reuses the equal search from results.json, ~5 min
    python opponents_compute.py merge     # combines the three into results_opponents.json

opponents.ipynb shows the results. results.ipynb is not touched.
"""
import os
import sys

import pandas as pd

import model as M

SEEDS = {"weak": 1700, "equal": 700, "strong": 2700}   # equal = the seed compute.ipynb used


def singles(m, res, fg, best_x):
    """The single strategies of compute.ipynb (step 1), with their tuned parameters."""
    (x7, y7), (a8, b8) = res["top_xy"][0], res["top_nml"][0]
    return {
        "1. Real coaches": M.coach,
        "2. Never": M.strategy([], fg),
        "3. Always": M.always,
        "4. Fixed rate (q = 0.2)": M.fixed_rate(0.2, fg),
        f"5. Short yardage (X = {best_x})": M.strategy([M.short_yardage(best_x)], fg),
        "6. Past midfield": M.strategy([M.past_midfield()], fg),
        f"7. Short and close (X, Y = {x7}, {y7})": M.strategy([M.short_and_close(x7, y7)], fg),
        f"8. No man's land ({a8}–{b8}, ≤10 yd)": M.strategy([M.no_mans_land(a8, b8, 10)], fg),
        "9. Goal line": M.strategy([M.goal_line()], fg),
        f"10. Trailing late (N = {res['best_n']})": M.strategy([M.trailing_late(res["best_n"])], fg),
        "11. Score-aware FG, last 3 drives": M.strategy([], fg, fg11=True),
        "12. Protect a lead on top of coaches": M.strategy([], fg, fallback="coach", protect=3),
    }


def run_opponent(o):
    m = M.estimate(M.load_data())
    res = M.load_results("results.json")
    fg, seed = res["fg_range"], SEEDS[o]
    best_x = int(res["best_x_opp"][o])                      # best "go if ≤ X yards" against this opponent
    out = {"opp": o, "best_x": best_x}

    # Step 1: each single strategy against this opponent (30,000 paired games)
    sing = M.compare(m, singles(m, res, fg, best_x), o, n=30_000, seed=seed + 100)
    out["singles"] = sing
    out["best_single"] = sing.drop("1. Real coaches")["diff (pp)"].idxmax()

    # Step 2: combination search (same candidates as the equal run)
    cands = M.candidates(res["top_xy"], res["top_nml"])
    if os.environ.get("SMOKE"):                           # quick test: tiny search
        cands = cands[:40]
    if o == "equal":                                      # already done in compute.ipynb
        final_names = [n for n in res["final"].index if n not in ("1. Real coaches",) and not n.startswith("Best single")]
        top = []
        for name in final_names:
            top += [c for c in cands if M.label(c) == name][:1]
        top = top or [res["best"]]
    else:
        history, _ = M.successive_halving(m, cands, fg, n=500, seed=seed, opp=o)
        top = [cands[i] for i in sorted(history[-1], key=history[-1].get, reverse=True)[:5]]

    # Final check on fresh paired games: top 5, the equal-opponent winner, the best single rule
    pols = {M.label(c): M.build(c, fg) for c in top}
    pols.setdefault(M.label(res["best"]), M.build(res["best"], fg))
    pols[f"Best single rule (X = {best_x})"] = M.strategy([M.short_yardage(best_x)], fg)
    final = M.compare(m, pols, o, n=50_000, seed=seed + 200)
    out["final"] = final
    combos = final.drop(["1. Real coaches", f"Best single rule (X = {best_x})"], errors="ignore")
    out["equal_winner"] = M.label(res["best"])
    if o == "equal":                                      # same winner as compute.ipynb
        out["best"] = res["best"]
    else:                                                 # the best combination in this final check
        out["best"] = next(c for c in top + [res["best"]] if M.label(c) == combos["diff (pp)"].idxmax())
    out["best_name"] = M.label(out["best"])
    best_name = out["best_name"]

    # Handbook of this opponent's best combination (tied game, not late)
    hb = M.handbook(m, M.build(out["best"], fg))
    out["handbook"] = hb
    M.save_results(out, f"results_opp_{o}{'_smoke' if os.environ.get('SMOKE') else ''}.json")
    print(final.round(3).to_string())
    print("Best combination:", best_name)


def merge():
    allres = {o: M.load_results(f"results_opp_{o}.json") for o in ["weak", "equal", "strong"]}
    flat = {}
    for o, r in allres.items():
        for k, v in r.items():
            flat[f"{o}/{k}"] = v
    M.save_results(flat, "results_opponents.json")
    print("Saved results_opponents.json")


if __name__ == "__main__":
    arg = sys.argv[1]
    merge() if arg == "merge" else run_opponent(arg)
