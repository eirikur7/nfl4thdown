# Pending changes (not yet in the main analysis)

Saved 21 Oct 2026. These are tested, but `compute.ipynb` / `results.ipynb` on `main` do **not** use them yet.
They go into the next full rerun, together with any other changes.

## 1. Punt-return touchdowns and muffed punts (code done on this branch)

`model.punt()` now also draws, from the same real punts near the spot:

| Outcome | Real punts | Share | In the simulator |
|---|---|---|---|
| Punt-return touchdown | 83 | ≈ 0.5% | the returning team scores (TD + extra point) |
| Muff / fumbled return lost | 171 | ≈ 1% | the kicking team gets the ball back at the spot |
| Muff recovered in the end zone | 6 | ≈ 0.04% | the kicking team scores |

Before, a return TD put the returner at the 1-yard line, and muffs were ignored.
From our own 25: 98.3% normal punts, 1.0% muffs, 0.7% return TDs.

Punts were already drawn from real punts kicked within ±5 yards of the same spot. So the following already depended on field position:
- touchbacks
- fair catches and out of bounds (a 0-yard return)
- return length

| Punting from | Gross yd | Touchback | Returned | Net |
|---|---|---|---|---|
| Own 1–9 | 49.7 | 0% | 62% | 42.5 |
| Own 20–29 | 49.3 | 1% | 50% | 43.6 |
| Own 40–49 | 44.7 | 14% | 20% | 40.1 |
| Midfield–opp 41 | 37.6 | 16% | 5% | 33.9 |
| Opp 40 and closer | 32.3 | 23% | 3% | 27.4 |

**Effect:** strategy-7 heatmap re-run only (`pending/heat_only.py`): `pending/heat_strategy7_with_punt_returns.png` / `.csv`.
- Everything is about 0.1–0.3 points lower.
- The top 10 cells are the same; only the ranking changed slightly: (6, 80) moved up past (5, 90).
- The best cell is still X = 6 at Y = 95 or 100 (54.4%).

## 2. New rule 13: leading late, don't go for it in your own half

When leading with ≤ N drives left and pos ≥ Z (on our own side), never go for it. Kick instead: a field goal if in range, else a punt.
A failed attempt there hands the opponent a short field for the game-winning score.

**Quick test** (`pending/lead_protect_test.py`):
- Added on top of the current best strategy (equal opponent).
- 50,000 paired games, seed 4000.
- Each cell is the gain over the best strategy without the rule, in pp with a 95% CI.

| | pos ≥ 50 | pos ≥ 60 | pos ≥ 70 | pos ≥ 80 |
|---|---|---|---|---|
| N = 1 | +0.17 [0.11, 0.23] | +0.14 [0.09, 0.20] | +0.06 [0.03, 0.10] | +0.03 [0.01, 0.06] |
| N = 2 | **+0.36 [0.25, 0.48]** | +0.27 [0.17, 0.37] | +0.12 [0.05, 0.18] | +0.04 [0.00, 0.09] |
| N = 3 | +0.34 [0.19, 0.49] | +0.28 [0.15, 0.41] | +0.17 [0.08, 0.26] | +0.06 [0.01, 0.12] |

- All 12 versions are significantly better. The best is N = 2, whole own half.
- That is more than the whole combination currently adds over the best single rule (+0.2).
- **Plan for the search:** rule 13 with N ∈ {1, 2, 3} and Z ∈ {40, 50, 60}. It overrides the go rules, like rule 12.

## 3. Rules tested individually in step 1 (current setup)

| # | Rule | Tuned in step 1 | In successive halving |
|---|---|---|---|
| 1 | Real coaches | baseline | – |
| 2 | Never go | – | no |
| 3 | Always go | – | no |
| 4 | Go at random with prob. q | q = 0–1 | no |
| 5 | Go if ≤ X yards | X = 0–15 → 6 | yes, as Y = 100 in rule 7 |
| – | Field-goal range | K = 45–68 yd → 57 | fixed at 57 |
| 6 | Past midfield | – | no (always hurt) |
| 7 | Go if ≤ X yards and pos ≤ Y | X 1–6 × Y 20–100 (5-yd steps) | top 10 cells |
| 8 | No man's land: A ≤ pos ≤ B and ≤ T yd | A × B, T = 10 | top 10 ranges, T = 6 or 10 |
| 9 | Goal line (pos ≤ 5) | – | on / off |
| 10 | Trailing late (≤ N drives) | N = 1–12 | N = 2, 3 or 6 |
| 11 | Score-aware FG (kick if it changes scores, else go) | – | no (covered by 12) |
| 12 | Score-changing FG (always kick if it changes scores, ≤ N drives) | – | N = 1, 2 or 3 |
| 13 | **New:** leading late, don't go in own half | – | proposed N = 1–3, Z = 40/50/60 |

When no rule says go, every candidate kicks: a field goal up to 57 yards, otherwise a punt.

**Search size:**
- Currently 7,392 candidates, about 35 min for the main run.
- With rule 13 it is about 3× larger, about 1 h, plus about 1 h for the weak/strong searches.
