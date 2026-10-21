"""4th down model: data, probabilities, game simulator, strategies and handbook.

Used by compute.ipynb (runs the simulations) and results.ipynb (shows the results).
`m` below is a plain dict with every probability estimated from the data (made by `estimate`).
A policy is a function  policy(m, togo, pos, diff, left, rng) -> "go" / "fg" / "punt".
  togo = yards to go, pos = yards to the opponent's end zone,
  diff = our score minus theirs, left = drives we have left, rng = random.Random
"""
import json
import math
import random
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

SEASONS = range(2018, 2026)
URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{}.csv.gz"
COLS = ["game_id", "season", "season_type", "qtr", "half_seconds_remaining", "posteam", "down", "ydstogo",
        "yardline_100", "play_type", "yards_gained", "first_down", "touchdown", "epa",
        "field_goal_result", "kick_distance", "extra_point_result", "return_yards",
        "interception", "fumble_lost", "incomplete_pass", "fixed_drive", "fixed_drive_result"]

MAX_FG = 68        # longest field goal allowed (NFL record)
KICKOFF = 70       # after a score the other team starts at pos 70
LATE = 3           # "late in the game" = 3 or fewer drives left (≈ 4th quarter)
SPEED = 1.0        # scales the number of simulated games (1.0 ≈ 30 min for compute.ipynb, 0.2 ≈ 6 min)

TOGO_BINS = [0, 1, 2, 3, 5, 7, 10, 99]
TOGO_LABELS = ["1", "2", "3", "4-5", "6-7", "8-10", "11+"]


def togo_bin(togo):
    """Index into TOGO_LABELS."""
    return next(i for i in range(7) if togo <= TOGO_BINS[i + 1])


def pos_bin(pos):
    """10-yard zones: 0 = pos 1-10, ..., 9 = pos 91-99."""
    return min(9, (pos - 1) // 10)


# ---------------------------------------------------------------- 1. Data

def load_data(folder="data"):
    """Regular-season play-by-play 2018-2025, only the columns we use.
    Reads data/pbp_<year>.csv.gz (downloads it if missing) and caches the result in data/pbp.pkl."""
    folder = Path(folder)
    cache = folder / "pbp.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    frames = []
    for y in SEASONS:
        local = folder / f"pbp_{y}.csv.gz"
        frames.append(pd.read_csv(local if local.exists() else URL.format(y), usecols=COLS, low_memory=False))
    pbp = pd.concat(frames, ignore_index=True)
    pbp = pbp[pbp.season_type == "REG"].reset_index(drop=True)
    # neutral = quarters 1-3, not the last 2 minutes of the first half
    pbp["neutral"] = (pbp.qtr <= 3) & ~((pbp.qtr == 2) & (pbp.half_seconds_remaining <= 120))
    folder.mkdir(exist_ok=True)
    pbp.to_pickle(cache)
    return pbp


# ---------------------------------------------------------------- 2. Probabilities

def play_model(plays):
    """(down, yards to go) -> (list of yards gained on real plays, P(turnover)). Down 3 = 3rd and 4th down."""
    return {(d, t): (g.loc[~g.to, "yards_gained"].astype(int).tolist(), g.to.mean())
            for (d, t), g in plays.groupby(["d", "t"])}


def fg_make_prob(x, b0, b1):
    """P(field goal made) from distance x: logistic fit p(x) = 1 / (1 + exp(-(b0 + b1 x)))."""
    return 1 / (1 + math.exp(-(b0 + b1 * x)))


def estimate(pbp):
    """Estimate every probability the simulator needs. Returns the dict m."""
    m = {}

    # Yards per play: real plays per (down, yards to go); 3rd and 4th down pooled, 11 = 11+ yards
    plays = pbp[pbp.neutral & pbp.play_type.isin(["run", "pass"]) & pbp.down.notna()].copy()
    plays["d"] = plays.down.clip(upper=3).astype(int)
    plays["t"] = plays.ydstogo.clip(upper=11).astype(int)
    plays["to"] = (plays.interception == 1) | (plays.fumble_lost == 1)
    m["plays"] = plays
    m["yards"] = {"equal": play_model(plays)}

    # P(1st down | yards to go), from 3rd and 4th downs
    conv = plays[(plays.d == 3) & (plays.ydstogo <= 15)].copy()
    conv["success"] = (conv.first_down == 1) | (conv.touchdown == 1)
    m["p_d"] = conv.groupby("ydstogo").success.agg(["mean", "count"])

    # Field goals: logistic regression by maximum likelihood (make rate per distance kept for the plot)
    fg = pbp[(pbp.play_type == "field_goal") & pbp.kick_distance.notna()]
    x, y = fg.kick_distance.to_numpy(float), (fg.field_goal_result == "made").to_numpy(float)
    b0, b1 = minimize(lambda b: np.sum(np.logaddexp(0, b[0] + b[1] * x) - y * (b[0] + b[1] * x)), [0, 0]).x
    fg_by_dist = fg.groupby(fg.kick_distance.astype(int)).field_goal_result.agg(
        made=lambda r: (r == "made").sum(), n="size")
    m["fg_logit"], m["fg_by_dist"], m["n_fg"] = (b0, b1), fg_by_dist, len(fg)
    m["p_fg"] = [0.0] + [fg_make_prob(pos + 18, b0, b1) if pos + 18 <= MAX_FG else 0.0
                         for pos in range(1, 100)]          # index = pos; kick distance = pos + 18

    # Punts (distance, return), extra points, drives per game.
    # Punters aim differently by field position (pinning deep near midfield), so a punt from pos is drawn
    # only from real punts kicked within ±5 yards of pos (window widened until it holds 200 punts).
    pu = pbp[(pbp.play_type == "punt") & pbp.yardline_100.notna()]
    m["punts"] = list(zip(pu.kick_distance.fillna(0).astype(int), pu.return_yards.fillna(0).astype(int)))
    yl, kd, ret = pu.yardline_100.to_numpy(int), pu.kick_distance.fillna(0).to_numpy(int), pu.return_yards.fillna(0).to_numpy(int)
    m["punts_at"] = [None]
    for pos in range(1, 100):
        w = 5
        while np.sum(np.abs(yl - pos) <= w) < 200:
            w += 1
        near = np.abs(yl - pos) <= w
        m["punts_at"].append(list(zip(kd[near], ret[near])))
    m["p_xp"] = (pbp[pbp.play_type == "extra_point"].extra_point_result == "good").mean()
    drives = pbp.dropna(subset=["posteam", "fixed_drive"]).groupby(["game_id", "posteam"]).fixed_drive.nunique()
    m["drives_mean"], m["drives"] = drives.mean(), int(round(drives.mean()))

    # Real coaches: P(go | yards-to-go bin, 10-yard zone), shrunk to the overall rate with 2 pseudo-observations
    fourth = pbp[pbp.neutral & (pbp.down == 4) & pbp.play_type.isin(["run", "pass", "punt", "field_goal"])].copy()
    fourth["go"] = fourth.play_type.isin(["run", "pass"])
    fourth["tb"] = pd.cut(fourth.ydstogo, TOGO_BINS, labels=False)
    fourth["pb"] = ((fourth.yardline_100 - 1) // 10).clip(upper=9).astype(int)
    r0 = fourth.go.mean()
    g = fourth.groupby(["tb", "pb"]).go.agg(["sum", "count"])
    m["p_go"] = {(tb, pb): r0 for tb in range(7) for pb in range(10)}     # fallback for cells never seen
    m["p_go"].update(((g["sum"] + 2 * r0) / (g["count"] + 2)).to_dict())
    m["go_table"] = (100 * (g["sum"] + 2 * r0) / (g["count"] + 2)).unstack().reindex(index=range(7), columns=range(10))
    m["go_rate"], m["n_fourth"] = r0, len(fourth)
    # When coaches kick: share of field goals (vs punts) at each pos, ±2 yards
    kicks = fourth[~fourth.go]
    share = kicks.groupby("yardline_100").play_type.agg(fgs=lambda r: (r == "field_goal").sum(), n="size")
    m["p_fg_try"] = [0.0] + [(lambda w: w.fgs.sum() / w.n.sum() if w.n.sum() else 0.0)(share.loc[p - 2:p + 2])
                             for p in range(1, 100)]

    # Opponent strength: plays only from the top / bottom third of teams by EPA per play (per season)
    epa = plays.groupby(["season", "posteam"]).epa.mean()
    third = epa.groupby(level=0).transform(lambda s: pd.qcut(s.rank(method="first"), 3,
                                                             labels=["weak", "middle", "strong"]))
    plays = plays.join(third.rename("strength"), on=["season", "posteam"])
    m["yards"]["strong"] = play_model(plays[plays.strength == "strong"])
    m["yards"]["weak"] = play_model(plays[plays.strength == "weak"])
    return m


# ---------------------------------------------------------------- 3. Simulator

def punt(m, pos, rng):
    """Opponent's starting pos after our punt from pos (touchback -> 80), drawn from real punts near pos."""
    punts = m["punts_at"][pos]
    kd, ret = punts[int(rng.random() * len(punts))]
    land = pos - kd
    return 80 if land <= 0 else min(99, max(1, 100 - land - ret))   # a return moves the receiver forward


def drive(m, pos, policy, yards, diff, left, rng, log=None):
    """One drive from 1st & 10 at pos (a Markov chain on (down, togo, pos)).
    Returns (our points, their points, opponent's start pos, how the drive ended)."""
    down, togo = 1, min(10, pos)
    while True:
        if down == 4:
            a = policy(m, togo, pos, diff, left, rng)
            if log is not None:
                log.append(a)
            if a == "fg":
                if rng.random() < m["p_fg"][pos]:
                    return 3, 0, KICKOFF, "Field goal"
                return 0, 0, min(93 - pos, 80), "Missed field goal"
            if a == "punt":
                return 0, 0, punt(m, pos, rng), "Punt"
        ys, p_to = yards[(min(down, 3), min(togo, 11))]
        if rng.random() < p_to:
            return 0, 0, max(1, 100 - pos), "Turnover"
        y = ys[int(rng.random() * len(ys))]          # draw one real play
        pos -= y
        if pos <= 0:
            return 6 + (rng.random() < m["p_xp"]), 0, KICKOFF, "Touchdown"
        if pos >= 100:
            return 0, 2, 60, "Safety"
        if y >= togo:
            down, togo = 1, min(10, pos)
        elif down == 4:
            return 0, 0, 100 - pos, "Turnover on downs"
        else:
            down, togo = down + 1, togo - y


def game(m, policy, opp, rng, log=None, ends=None):
    """One game: m["drives"] drives per team, our policy vs a real-coach opponent. Returns [our score, theirs]."""
    n = m["drives"]
    score, left = [0, 0], [n, n]
    policies, yards = (policy, coach), (m["yards"]["equal"], m["yards"][opp])
    team, pos = int(rng.random() < 0.5), KICKOFF
    for _ in range(2 * n):
        pf, pa, pos, end = drive(m, pos, policies[team], yards[team], score[team] - score[1 - team],
                                 left[team], rng, log if team == 0 else None)
        if ends is not None and team == 0:
            ends.append(end)
        score[team] += pf
        score[1 - team] += pa
        left[team] -= 1
        team = 1 - team
    return score


def n_games(n):
    return max(500, int(n * SPEED))


def run(m, policy, opp="equal", n=10_000, seed=0):
    """Play n games (game i always uses the same random numbers -> paired comparisons).
    Returns (result per game: 1 win, 0.5 tie, 0 loss;  share of our 4th downs where we went for it)."""
    n = n_games(n)
    out, log = np.empty(n), []
    for i in range(n):
        a, b = game(m, policy, opp, random.Random(seed * 1_000_003 + i), log)
        out[i] = 1.0 if a > b else (0.5 if a == b else 0.0)
    return out, log.count("go") / max(1, len(log))


def ci(x):
    """Mean and 95% confidence interval (CLT)."""
    mean, se = x.mean(), x.std(ddof=1) / math.sqrt(len(x))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def compare(m, policies, opp="equal", n=30_000, seed=100):
    """Each policy vs real coaches on the same games: win prob. and paired difference in pp."""
    base = run(m, coach, opp, n, seed)[0]
    rows = []
    for name, pol in policies.items():
        out, go = run(m, pol, opp, n, seed)
        p, _, _ = ci(out)
        d, dlo, dhi = ci(100 * (out - base))
        rows.append({"strategy": name, "p̂": p, "diff (pp)": d, "diff lower": dlo, "diff upper": dhi, "go %": 100 * go})
    return pd.DataFrame(rows).set_index("strategy")


# ---------------------------------------------------------------- 4. Strategies

def coach(m, togo, pos, diff, left, rng):
    """Real coaches: go with the probability coaches go in this cell, otherwise kick like coaches."""
    if rng.random() < m["p_go"][(togo_bin(togo), pos_bin(pos))]:
        return "go"
    return "fg" if rng.random() < m["p_fg_try"][pos] else "punt"


def always(m, togo, pos, diff, left, rng):
    return "go"


def kick(pos, fg_range):
    """Our kick: field goal if pos <= fg_range, otherwise punt."""
    return "fg" if pos <= fg_range else "punt"


def fixed_rate(q, fg_range):
    """Strategy 4: go with probability q anywhere, otherwise kick."""
    return lambda m, togo, pos, diff, left, rng: "go" if rng.random() < q else kick(pos, fg_range)


# Go rules: (togo, pos, diff, left) -> True if we go for it
def short_yardage(X):     return lambda togo, pos, diff, left: togo <= X                # rule 5
def past_midfield():      return lambda togo, pos, diff, left: pos < 50                 # rule 6
def short_and_close(X, Y): return lambda togo, pos, diff, left: togo <= X and pos <= Y  # rule 7
def no_mans_land(A, B, T=99): return lambda togo, pos, diff, left: A <= pos <= B and togo <= T  # rule 8
def goal_line():          return lambda togo, pos, diff, left: pos <= 5                 # rule 9
def trailing_late(N):     return lambda togo, pos, diff, left: diff < 0 and left <= N   # rule 10


def fg_changes_scores(diff):
    """Rule 11: does a field goal change how many scores separate the teams?"""
    return 6 <= diff <= 8 or -11 <= diff <= -9


def score_changing_fg(N, pos, fg_range, diff, left):
    """Rule 12: with N or fewer drives left, in field-goal range, a field goal changes how many scores separate
    the teams (leading by 6-8 -> 9+, two scores; trailing by 9-11 -> within 8, one score)."""
    return N is not None and left <= N and pos <= fg_range and fg_changes_scores(diff)


def strategy(go_rules=(), fg_range=37, fg11=False, fallback="kick", protect=None):
    """Go if ANY go rule says so. fg11 = rule 11 (score-aware field goal, late only).
    protect = N: rule 12, with <= N drives left always kick the field goal when it changes the number of scores
    between the teams (lead 6-8 -> 9+, or trail 9-11 -> within 8), overriding every go rule.
    fallback when no rule says go: 'kick' (FG if in range, else punt) or 'coach' (decide like a real coach)."""
    def policy(m, togo, pos, diff, left, rng):
        if score_changing_fg(protect, pos, fg_range, diff, left):
            return "fg"
        if any(rule(togo, pos, diff, left) for rule in go_rules):
            return "go"
        if fg11 and pos <= fg_range and left <= LATE:
            return "fg" if fg_changes_scores(diff) else "go"
        return coach(m, togo, pos, diff, left, rng) if fallback == "coach" else kick(pos, fg_range)
    return policy


# A candidate combination is a dict:
#   yt = None / ["5", X] / ["7", [X, Y]]   (yards-to-go rule),   r6, r9, r11 = True/False,
#   r8 = None / [A, B, T],   r10 = None / N,   r12 = None / N,   fallback = "kick" / "coach"

def build(c, fg_range):
    """Turn a candidate dict into a policy."""
    rules = []
    if c["yt"] and c["yt"][0] == "5": rules.append(short_yardage(c["yt"][1]))
    if c["yt"] and c["yt"][0] == "7": rules.append(short_and_close(*c["yt"][1]))
    if c["r6"]: rules.append(past_midfield())
    if c["r8"]: rules.append(no_mans_land(*c["r8"]))
    if c["r9"]: rules.append(goal_line())
    if c["r10"]: rules.append(trailing_late(c["r10"]))
    return strategy(rules, fg_range, fg11=c["r11"], fallback=c["fallback"], protect=c.get("r12"))


def label(c):
    """Readable name, e.g. '≤4 yd + trailing (≤2 drives) | else kick'."""
    p = []
    if c["yt"]:
        if c["yt"][0] == "5" or c["yt"][1][1] >= 100:
            p.append(f"≤{c['yt'][1] if c['yt'][0] == '5' else c['yt'][1][0]} yd")
        else:
            p.append(f"≤{c['yt'][1][0]} yd & pos ≤{c['yt'][1][1]}")
    if c["r6"]: p.append("past midfield")
    if c["r8"]: p.append(f"no man's land {c['r8'][0]}–{c['r8'][1]} (≤{c['r8'][2]} yd)")
    if c["r9"]: p.append("goal line")
    if c["r10"]: p.append(f"trailing (≤{c['r10']} drives)")
    if c["r11"]: p.append("score-aware FG")
    if c.get("r12"): p.append(f"score-changing FG (≤{c['r12']} drives)")
    return (" + ".join(p) or "no rules") + f" | else {c['fallback']}"


def candidates(top_xy, top_nml, nml_togo=(6, 10)):
    """All combinations searched in step 2. The yards-to-go rule is one of the top strategy-7 cells
    (X, Y), where Y = 100 is plain "go if <= X yards"; no man's land is one of the top (A, B) ranges.
    Rule 6 (past midfield) is left out: it always lowered the win probability. Rule 11 is left out: its kicking
    half is rule 12, and its "otherwise go in field-goal range late" half only lowers it. Every candidate kicks
    (field goal in range, else punt) when no rule says go; "else do what coaches do" is not searched."""
    yt = [None] + [["7", [int(x), int(y)]] for x, y in top_xy]
    r8 = [None] + [[a, b, t] for a, b in top_nml for t in nml_togo]
    return [dict(yt=a, r6=b, r8=c, r9=d, r10=e, r11=f, r12=h, fallback=g)
            for a, b, c, d, e, f, h, g in product(yt, [False], r8, [False, True], [None, 2, 3, 6],
                                                  [False], [None, 1, 2, 3], ["kick"])]


def size(c):
    """Number of rules in a candidate."""
    return bool(c["yt"]) + c["r6"] + bool(c["r8"]) + c["r9"] + bool(c["r10"]) + c["r11"] + bool(c.get("r12"))


_SH = {}


def _sh_run(i):
    """Worker for successive_halving: one candidate on the current round's games."""
    out, go = run(_SH["m"], build(_SH["cands"][i], _SH["fg"]), _SH["opp"], n=_SH["n"], seed=_SH["seed"])
    return out, go


def successive_halving(m, cands, fg_range, rounds=4, n=1_000, seed=700, opp="equal", workers=None):
    """Play all candidates on n games, keep the best quarter, play them on 4x more games, repeat.
    Candidates with identical results (same decisions) are merged, keeping the simplest.
    Runs the candidates in parallel (same results as one process: every candidate uses the same seed).
    Returns (list of {candidate index: diff in pp} per round, list of game counts)."""
    import multiprocessing as mp
    workers = workers or mp.cpu_count()
    alive = sorted(range(len(cands)), key=lambda i: size(cands[i]))
    history, games = [], []
    for r in range(rounds):
        base = run(m, coach, opp, n=n, seed=seed + r)[0]
        _SH.update(m=m, cands=cands, fg=fg_range, opp=opp, n=n, seed=seed + r)
        with mp.get_context("fork").Pool(workers) as pool:
            results = pool.map(_sh_run, alive, chunksize=max(1, len(alive) // (8 * workers)))
        scores, seen = {}, set()
        for i, (out, go) in zip(alive, results):
            key = (out.tobytes(), round(go, 9))
            if key in seen:
                continue
            seen.add(key)
            scores[i] = 100 * (out - base).mean()
        history.append(scores)
        games.append(n_games(n))
        print(f"Round {r + 1}: {len(scores)} distinct candidates on {n_games(n):,} games each")
        alive = sorted(scores, key=scores.get, reverse=True)[:max(5, len(scores) // 4)]
        n *= 4
    return history, games


RULE_NAMES = {"5": "5. Short yardage", "7": "7. Short and close", "6": "6. Past midfield",
              "8": "8. No man's land", "9": "9. Goal line", "10": "10. Trailing late", "11": "11. Score-aware FG (late)",
              "12": "12. Score-changing FG (late)"}


def rule_effects(cands, scores):
    """Linear regression of each candidate's gain (pp) on 0/1 indicators of its rules, one model per fallback.
    Returns a table with β (average effect of adding the rule) and its standard error, and R² per fallback."""
    rows, r2 = [], {}
    for fb in dict.fromkeys(cands[i]["fallback"] for i in scores):
        ids = [i for i in scores if cands[i]["fallback"] == fb]
        c = [cands[i] for i in ids]
        X = np.array([[1,
                       bool(k["yt"]) and k["yt"][0] == "5", bool(k["yt"]) and k["yt"][0] == "7",
                       k["r6"], bool(k["r8"]), k["r9"], bool(k["r10"]), k["r11"], bool(k.get("r12"))]
                      for k in c], dtype=float)
        y = np.array([scores[i] for i in ids])
        keys = np.array(["", "5", "7", "6", "8", "9", "10", "11", "12"])
        used = [0] + [j for j in range(1, X.shape[1]) if X[:, j].min() != X[:, j].max()]   # drop rules never varied
        X, keys = X[:, used], keys[used]
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ beta
        se = np.sqrt(np.diag(res @ res / (len(y) - X.shape[1]) * np.linalg.inv(X.T @ X)))
        r2[fb] = 1 - res @ res / np.sum((y - y.mean()) ** 2)
        for k, key in enumerate(keys[1:], start=1):
            rows.append({"rule": RULE_NAMES[key], "fallback": fb, "β": beta[k], "se": se[k]})
    return pd.DataFrame(rows), r2


# ---------------------------------------------------------------- 5. Handbook

def handbook(m, policy, max_togo=15):
    """For every 4th down (yards to go, pos): the call the policy makes in a normal situation
    (tied game, plenty of drives left), next to the probabilities behind it and what coaches do."""
    rng = random.Random(0)
    rows = []
    for togo, pos in product(range(1, max_togo + 1), range(1, 100)):
        if togo > pos:                     # impossible: the line to gain is past the goal line
            continue
        cell = (togo_bin(togo), pos_bin(pos))
        coach_call = "go" if m["p_go"][cell] > 0.5 else ("fg" if m["p_fg_try"][pos] > 0.5 else "punt")
        rows.append({"togo": togo, "pos": pos, "call": policy(m, togo, pos, 0, m["drives"], rng),
                     "P(1st down)": m["p_d"]["mean"].get(togo, np.nan),
                     "FG distance": pos + 18, "P(FG good)": m["p_fg"][pos],
                     "coaches go %": 100 * m["p_go"][cell], "coaches usually": coach_call})
    return pd.DataFrame(rows)


def go_thresholds(hb):
    """Handbook in one table: per 10-yard zone, go for it with this many yards to go or fewer."""
    hb = hb.assign(zone=((hb.pos - 1) // 10) * 10 + 1)
    go = hb[hb.call == "go"].groupby("zone").togo.max()
    kick_ = hb[hb.call != "go"].groupby("zone").call.agg(lambda s: "/".join(sorted(set(s))))
    out = pd.DataFrame({"go if yards to go ≤": go, "otherwise": kick_})
    out["go if yards to go ≤"] = out["go if yards to go ≤"].map(lambda v: "never" if pd.isna(v) else int(v))
    out.index = [f"{z}-{min(z + 9, 99)}" for z in out.index]
    out.index.name = "pos (yards to end zone)"
    return out


# ---------------------------------------------------------------- 6. Save / load results

def save_results(res, path="results.json"):
    """Save a dict of numbers, lists and DataFrames as JSON."""
    def enc(v):
        if isinstance(v, pd.Series):
            v = v.to_frame()
        if isinstance(v, pd.DataFrame):
            return {"table": v.to_dict(orient="split")}
        return v
    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: enc(v) for k, v in res.items()}, f, ensure_ascii=False, indent=1,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))


def load_results(path="results.json"):
    def dec(v):
        if isinstance(v, dict) and set(v) == {"table"}:
            t = v["table"]
            return pd.DataFrame(t["data"], index=t["index"], columns=t["columns"])
        return v
    with open(path, encoding="utf-8") as f:
        return {k: dec(v) for k, v in json.load(f).items()}


def coach_vs_coach_test(out):
    """Test H0: p = 0.5 for coach vs coach. Returns (p̂, lower, upper, p-value)."""
    p, lo, hi = ci(out)
    z = (p - 0.5) / (out.std() / math.sqrt(len(out)))
    return p, lo, hi, 2 * (1 - norm.cdf(abs(z)))
