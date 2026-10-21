"""Plots for opponents.ipynb (best strategies per opponent strength). Same style as plots.py."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

import model as M
from plots import BLUE, ORANGE, DARK, LIGHT, CALLS, CALL_COLORS

OPPS = ["weak", "equal", "strong"]
OPP_COLORS = {"weak": LIGHT, "equal": BLUE, "strong": DARK}


def summary(r):
    """Gain over coaches (same opponent) for the best single rule, the equal-opponent winner and the
    opponent's own best combination, with 95% CIs."""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    rows = [("Best single rule", lambda o: f"Best single rule (X = {r[f'{o}/best_x']})"),
            ("Best combination vs equal opponent", lambda o: r[f"{o}/equal_winner"]),
            ("Best combination for this opponent", lambda o: r[f"{o}/best_name"])]
    for j, o in enumerate(OPPS):
        f = r[f"{o}/final"]
        for i, (lab, key) in enumerate(rows):
            row = f.loc[key(o)]
            y = i + (j - 1) * 0.22
            ax.errorbar(row["diff (pp)"], y, xerr=[[row["diff (pp)"] - row["diff lower"]], [row["diff upper"] - row["diff (pp)"]]],
                        fmt="o", color=OPP_COLORS[o], ecolor=OPP_COLORS[o], capsize=3, ms=7,
                        label=f"Opponent: {o}" if i == 0 else None)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([lab for lab, _ in rows])
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Gain over coaches, same opponent (pp, 95% CI, 100,000 paired games)")
    ax.set_title("Best strategies per opponent strength")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def _handbook(m, cand, fg):
    """Handbook of a candidate; where it falls back to the coaches, show their MOST COMMON call
    (as in the "Real coaches" panel of results.ipynb) instead of a random draw."""
    most_common = lambda m_, togo, pos, diff, left, rng: (
        "go" if m_["p_go"][(M.togo_bin(togo), M.pos_bin(pos))] > 0.5 else ("fg" if m_["p_fg_try"][pos] > 0.5 else "punt"))
    orig, M.coach = M.coach, most_common
    try:
        return M.handbook(m, M.build(cand, fg))
    finally:
        M.coach = orig


def handbooks(r, m=None, fg=None, max_togo=10):
    """The best combination's handbook against each opponent strength, side by side.
    With m and fg, rebuilt so a coach fallback shows the coaches' most common call."""
    cmap = ListedColormap(CALL_COLORS)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.0), sharey=True)
    for ax, o in zip(axs, OPPS):
        hb = _handbook(m, r[f"{o}/best"], fg) if m is not None else r[f"{o}/handbook"]
        hb = hb[hb.togo <= max_togo]
        grid = hb.pivot(index="togo", columns="pos", values="call").reindex(columns=range(1, 100))
        codes = grid.apply(lambda s: s.map({c: i for i, c in enumerate(CALLS)})).values.astype(float)
        ax.imshow(codes, cmap=cmap, vmin=-0.5, vmax=2.5, aspect="auto", origin="lower", extent=(0, 99, 0, max_togo))
        ax.invert_xaxis()
        ax.set(xlabel="pos (yards to opponent's end zone)", title=f"{o.capitalize()} opponent")
        ax.set_yticks(range(1, max_togo + 1))
        ax.set_xticks(range(0, 100, 20))
        ax.grid(False)
    axs[0].set_ylabel("Yards to go")
    fig.legend(handles=[Patch(color=c, label=l) for c, l in zip(CALL_COLORS, ["Go for it", "Field goal", "Punt"])],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05))
    plt.tight_layout()
    plt.show()
