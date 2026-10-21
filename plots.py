"""All plots for results.ipynb. Each function takes the data dict m and/or the results dict res."""
import random
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.stats import norm

import model as M

plt.rcParams.update({"figure.figsize": (9, 4.8), "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GRAY, DARK, LIGHT = "#2a78d6", "#eb6834", "#8a8984", "#13202e", "#9ec5ef"


# ---------------------------------------------------------------- Data

def yards_per_play(m):
    ys = np.array(m["yards"]["equal"][(1, 10)][0])
    vals, cnt = np.unique(np.clip(ys, -10, 40), return_counts=True)
    plt.bar(vals, cnt / cnt.sum(), width=0.9, color=BLUE, label="Data")
    x = np.linspace(-10, 40, 300)
    plt.plot(x, norm.pdf(x, ys.mean(), ys.std()), color=ORANGE, lw=2, label="Normal, same mean and SD")
    plt.axvline(ys.mean(), color="black", ls="--", lw=1)
    plt.xlabel("Yards gained on 1st & 10 (40 = 40+)")
    plt.ylabel("Probability")
    plt.title(f"Yards per play: mean {ys.mean():.1f}, median {np.median(ys):.0f}")
    plt.legend()
    plt.show()


def field_goals(m):
    fg = m["fg_by_dist"]
    d = np.arange(18, M.MAX_FG + 1)
    b0, b1 = m["fg_logit"]
    plt.scatter(fg.index, fg.made / fg.n, s=np.sqrt(fg.n) * 2, color=ORANGE, alpha=0.6,
                label="Each distance (size = number of kicks)")
    plt.plot(d, [M.fg_make_prob(x, b0, b1) for x in d], color=BLUE, lw=2.5,
             label=f"Logistic fit (β₀ = {b0:.2f}, β₁ = {b1:.3f})")
    plt.xlim(17, 70)
    plt.ylim(0, 1.02)
    plt.xlabel("Kick distance (yards)")
    plt.ylabel("P(made)")
    plt.title(f"Field goals ({m['n_fg']:,} kicks)")
    plt.legend(loc="lower left")
    plt.show()


def coach_go_rates(m):
    t = m["go_table"].values
    fig, ax = plt.subplots(figsize=(9, 3.8))
    im = ax.imshow(t, cmap="Blues", aspect="auto", vmin=0, vmax=100)
    for i, j in product(range(7), range(10)):
        if not np.isnan(t[i, j]):
            ax.text(j, i, f"{t[i, j]:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if t[i, j] > 55 else "black")
    ax.set_xticks(range(10), [f"{10 * j + 1}-{10 * j + 10}" for j in range(10)])
    ax.set_yticks(range(7), M.TOGO_LABELS)
    ax.set(xlabel="pos (yards to opponent's end zone)", ylabel="Yards to go",
           title=f"Real coaches: % going for it ({m['n_fourth']:,} decisions, {m['go_rate']:.1%} overall)")
    ax.grid(False)
    fig.colorbar(im, label="% going for it")
    plt.show()


# ---------------------------------------------------------------- Simulator check

def validation(res):
    conv, ends = res["conversion"], res["drive_ends"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(conv.index, conv["mean"], "o-", color=BLUE, label="Data")
    axs[0].plot(conv.index, conv["simulated"], "^--", color=ORANGE, label="Simulator")
    axs[0].set(xlabel="Yards to go", ylabel="P(1st down)", ylim=(0, 0.8), title="Probability of a 1st down")
    axs[0].legend()
    y = np.arange(len(ends))
    axs[1].barh(y - 0.2, ends["Data"], 0.4, color=BLUE, label="Data")
    axs[1].barh(y + 0.2, ends["Simulator"], 0.4, color=ORANGE, label="Simulator")
    axs[1].set_yticks(y, ends.index)
    axs[1].invert_yaxis()
    axs[1].set(xlabel="Share of drives", title="How drives end")
    axs[1].legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------- Step 1

def fg_sweep(res):
    s = res["fg_sweep"]
    plt.fill_between(s.index, s.lo, s.hi, color=BLUE, alpha=0.2)
    plt.plot(s.index, s.p, "o-", color=BLUE)
    plt.xlabel("Longest field goal our team attempts, K (yards)")
    plt.ylabel("Win probability")
    plt.title("How far should we try field goals? (go if ≤ 4 yards, else kick)")
    plt.show()


def aggressiveness(res):
    for df, col, lab in [(res["sweep_q"], ORANGE, "4. Go with probability q"),
                         (res["sweep_x"], BLUE, "5. Go if yards to go ≤ X")]:
        plt.fill_between(100 * df.go, df.lo, df.hi, color=col, alpha=0.2)
        plt.plot(100 * df.go, df.p, "o-", color=col, label=lab)
    for _, r in res["sweep_x"].iterrows():
        plt.annotate(f"X={r.value:.0f}", (100 * r.go, r.p), textcoords="offset points", xytext=(0, 7),
                     ha="center", fontsize=8)
    go, p = res["coach_point"]
    plt.plot(100 * go, p, "kD", ms=8)
    plt.annotate("Real coaches", (100 * go, p), textcoords="offset points", xytext=(8, -12))
    plt.axhline(0.5, color=GRAY, ls=":")
    plt.xlabel("Share of 4th downs where the team goes for it (%)")
    plt.ylabel("Win probability")
    plt.title("Win probability vs aggressiveness (equal opponent)")
    plt.legend(loc="lower center")
    plt.show()


def heatmap(ax, table, fmt, vmin, vmax, bold=()):
    """Coloured table with a number in each cell; cells in `bold` get a box."""
    v = table.values.astype(float)
    im = ax.imshow(v, cmap="RdBu", vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
    for i, j in product(range(v.shape[0]), range(v.shape[1])):
        if np.isnan(v[i, j]):
            continue
        top = any(str(table.index[i]) == str(r) and str(table.columns[j]) == str(c) for r, c in bold)
        ax.text(j, i, fmt(v[i, j]), ha="center", va="center", fontsize=8, fontweight="bold" if top else "normal")
        if top:
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="black", lw=1.5))
    ax.set_xticks(range(v.shape[1]), table.columns)
    ax.set_yticks(range(v.shape[0]), table.index)
    ax.grid(False)
    return im


def short_and_close(res):
    fig, ax = plt.subplots(figsize=(13, 4.6))
    im = heatmap(ax, res["heat"], lambda v: f"{v:.3f}", 0.44, 0.56, res["top_xy"])
    ax.set(xlabel="Y: go only if pos ≤ Y (100 = anywhere)", ylabel="X: max yards to go",
           title="Strategy 7 (boxed: top 10, carried forward)")
    fig.colorbar(im, label="Win probability")
    plt.show()


def no_mans_land(m, res):
    fg_range, (a, b) = res["fg_range"], res["top_nml"][0]
    rng = random.Random(3)
    pos = np.arange(20, 71)
    net_punt = [np.mean([p - (100 - M.punt(m, p, rng)) for _ in range(4000)]) for p in pos]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.3))
    axs[0].plot(pos, [m["p_fg"][p] if p <= M.MAX_FG - 18 else np.nan for p in pos], color=BLUE, lw=2,
                label="P(field goal made)")
    axs[0].axvline(fg_range, color=BLUE, ls=":", lw=1)
    axs[0].text(fg_range - 0.5, 0.9, "our FG limit", color=BLUE, fontsize=8)
    axs[0].set(ylabel="P(field goal made)", ylim=(0, 1), xlabel="pos")
    axs[0].invert_xaxis()
    ax2 = axs[0].twinx()
    ax2.plot(pos, net_punt, color=ORANGE, lw=2, label="Expected net punt (yards)")
    ax2.set(ylabel="Expected net punt (yards)", ylim=(0, 50))
    ax2.grid(False)
    axs[0].axvspan(a, b, color=GRAY, alpha=0.2)
    axs[0].text((a + b) / 2, 0.06, "best no man's land", ha="center", fontsize=9)
    lines = [ln for ln in axs[0].get_lines() + ax2.get_lines() if not ln.get_label().startswith("_")]
    axs[0].legend(handles=lines, loc="lower left", fontsize=9)
    axs[0].set_title("Field goals get unlikely, punts gain little")
    im = heatmap(axs[1], 100 * (res["nml"] - res["nml_base"]), lambda v: f"{v:+.1f}", -2, 2, res["top_nml"])
    axs[1].set(xlabel="B (upper end of pos range)", ylabel="A (lower end)",
               title=f"Adding 'go if A ≤ pos ≤ B and ≤ 10 yards' to 'go if ≤ {res['best_x']} yards'")
    fig.colorbar(im, ax=axs[1], label="pp vs short yardage alone")
    plt.tight_layout()
    plt.show()


def diffs(df, title):
    """Each strategy's change in win probability vs real coaches, with 95% CI."""
    df = df.sort_values("diff (pp)")
    y = np.arange(len(df))
    cols = [BLUE if lo > 0 else (ORANGE if hi < 0 else GRAY) for lo, hi in zip(df["diff lower"], df["diff upper"])]
    plt.figure(figsize=(9, 0.42 * len(df) + 1.2))
    plt.errorbar(df["diff (pp)"], y, xerr=[df["diff (pp)"] - df["diff lower"], df["diff upper"] - df["diff (pp)"]],
                 fmt="none", ecolor=GRAY, capsize=3)
    plt.scatter(df["diff (pp)"], y, c=cols, s=45, zorder=3)
    plt.yticks(y, df.index)
    plt.axvline(0, color="black", lw=0.8)
    plt.xlabel("Win probability vs real coaches (percentage points, 95% CI)")
    plt.title(title)
    plt.show()


# ---------------------------------------------------------------- Step 2

def halving(res):
    rounds = len(res["halving"])
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for r, scores in enumerate(res["halving"]):
        jitter = np.random.default_rng(r).uniform(-0.25, 0.25, len(scores))
        ax.scatter(r + jitter, scores, s=6, color=BLUE if r == rounds - 1 else LIGHT, alpha=0.6)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(rounds), [f"round {r + 1}: {g:,} games" for r, g in enumerate(res["halving_games"])])
    ax.set(ylabel="Win prob. vs coaches (pp)", title="Successive halving: each round keeps the top quarter")
    plt.show()


def effects(res):
    e = res["effects"]
    names = list(dict.fromkeys(e.rule))
    y = np.arange(len(names))
    plt.figure(figsize=(9, 4.3))
    fbs = [fb for fb in ["kick", "coach"] if (e.fallback == fb).any()]
    w = 0.76 / len(fbs)
    for k, fb in enumerate(fbs):
        f = e[e.fallback == fb].set_index("rule").loc[names]
        plt.barh(y + (k - (len(fbs) - 1) / 2) * w, f["β"], w, xerr=1.96 * f["se"], color=ORANGE if fb == "kick" else BLUE,
                 label=f"fallback: {fb}", error_kw=dict(ecolor=GRAY, capsize=2))
    plt.yticks(y, names)
    plt.gca().invert_yaxis()
    plt.axvline(0, color="black", lw=0.8)
    plt.xlabel("Average effect of adding the rule (pp)")
    plt.title("Regression: effect of each rule")
    plt.legend()
    plt.show()


def short_label(name):
    """Compact strategy name for plots."""
    if name.startswith("1."):
        return "Real coaches"
    if name.startswith("Best single"):
        return "Best single rule: go if ≤ " + name.split("X = ")[1].rstrip(")") + " yd"
    for a, b in [(" | else kick", ""), (" | else coach", ", else coach"), (" yd inside own 30)", " inside own 30)"),
                 ("(not inside own 30)", "(never inside own 30)"), ("no man's land ", ""), (" (≤6 yd)", " ≤6 yd"),
                 (" (≤10 yd)", " ≤10 yd"), ("trailing (≤", "trailing late ("), ("protect lead (≤3 drives)", "protect lead"),
                 (" drives)", ")"), (" + ", " · ")]:
        name = name.replace(a, b)
    return name


def final_check(res, ax=None):
    """Final check with the chosen (refined) strategy highlighted."""
    df = res["final"].sort_values("diff (pp)")
    chosen = res["best_name"]
    show = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 0.42 * len(df) + 1.2))
    for i, (name, r) in enumerate(df.iterrows()):
        col = ORANGE if name == chosen else (GRAY if name.startswith("1.") else BLUE)
        ax.errorbar(r["diff (pp)"], i, xerr=[[r["diff (pp)"] - r["diff lower"]], [r["diff upper"] - r["diff (pp)"]]],
                    fmt="none", ecolor=GRAY, capsize=3)
        ax.scatter(r["diff (pp)"], i, color=col, s=70 if name == chosen else 45, zorder=3)
    ax.set_yticks(range(len(df)), [short_label(n) for n in df.index])
    for lab, name in zip(ax.get_yticklabels(), df.index):
        if name == chosen:
            lab.set_color(ORANGE)
            lab.set_fontweight("bold")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Win probability vs real coaches (pp, 95% CI)")
    ax.set_title("Final check: top combinations vs real coaches (orange = chosen)")
    if show:
        plt.show()


# ---------------------------------------------------------------- Step 3

def opponents(res):
    a = res["aggr"]
    x = range(len(a))
    for o, col, mk in zip(["weak", "equal", "strong"], [LIGHT, BLUE, DARK], "oDs"):
        if "aggr_lo" in res:                                # 95% CI bands
            plt.fill_between(x, res["aggr_lo"][o], res["aggr_hi"][o], color=col, alpha=0.18, lw=0)
        base = f" (coaches win {100 * res['aggr_base'][o]:.0f}%)" if "aggr_base" in res else ""
        plt.plot(x, a[o], marker=mk, color=col, label=f"Opponent: {o}{base}")
    plt.xticks(x, ["never"] + [f"≤{v}" for v in a.index[1:-1]] + ["always"])
    plt.axhline(0, color="black", lw=0.8)
    plt.xlabel("Go for it if yards to go ≤ X   (more aggressive →)")
    plt.ylabel("Gain over coaches, same opponent (pp)")
    plt.title("Gain in win probability over real coaches facing the same opponent (95% CI)")
    plt.legend()
    plt.show()


# ---------------------------------------------------------------- Handbook

CALLS = ["go", "fg", "punt"]
CALL_COLORS = [BLUE, ORANGE, "#d9d8d4"]


def handbook_map(hb, max_togo=10):
    """Two maps side by side: the handbook's call vs what coaches usually do."""
    hb = hb[hb.togo <= max_togo]
    cmap = ListedColormap(CALL_COLORS)
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.2), sharey=True)
    for ax, col, title in [(axs[0], "call", "Handbook (best strategy)"), (axs[1], "coaches usually", "Real coaches (most common call)")]:
        grid = hb.pivot(index="togo", columns="pos", values=col).reindex(columns=range(1, 100))
        codes = grid.apply(lambda s: s.map({c: i for i, c in enumerate(CALLS)})).values.astype(float)
        ax.imshow(codes, cmap=cmap, vmin=-0.5, vmax=2.5, aspect="auto", origin="lower",
                  extent=(0, 99, 0, max_togo))      # cell k spans (k-1, k]: "≤ 4" ends exactly at tick 4
        ax.invert_xaxis()
        ax.set(xlabel="pos (yards to opponent's end zone)", title=title)
        ax.set_yticks(range(1, max_togo + 1))
        ax.set_xticks(range(0, 100, 10))
        ax.grid(False)
    axs[0].set_ylabel("Yards to go")
    fig.legend(handles=[Patch(color=c, label=l) for c, l in zip(CALL_COLORS, ["Go for it", "Field goal", "Punt"])],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))
    plt.tight_layout()
    plt.show()
