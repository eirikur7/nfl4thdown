# 4th down: go for it, kick, or punt?

Applied Probability (T-811, Reykjavik University), project 1.
Which rule for when to go for it on 4th down wins the most games? We estimate probabilities from NFL
play-by-play data (2018–2025, nflverse), simulate whole games as a Markov chain, and compare strategies
with Monte Carlo. The handbook shows the best call for every 4th down.

**Open [`results.ipynb`](results.ipynb)** for all plots, tables and the handbook.

## Files

| File | What it does |
|---|---|
| `results.ipynb` | Shows everything. Reads `results.json`, runs in seconds. |
| `compute.ipynb` | Works through the data and runs all simulations, saves `results.json` (~30 min). |
| `model.py` | Data, probabilities, simulator, strategies, handbook. |
| `plots.py` | The plots used in `results.ipynb`. |
| `results.json` | Saved simulation results. |
| `data/` | Play-by-play files `pbp_<year>.csv.gz` (downloaded if missing) and the cache `pbp.pkl`. |

## Run

```bash
conda create -n nfl4th python numpy pandas scipy matplotlib notebook
conda activate nfl4th
```

Open `results.ipynb` and run all. To redo the simulations, run `compute.ipynb` first
(set `M.SPEED = 0.2` at the top for a quick ~6 min test run).

## Method in short

1. **Probabilities from data:** yards gained per play (sampled from real plays), P(1st down | yards to go),
   P(field goal good | distance) (data blended with a logistic regression), punts, and what coaches do on 4th down.
2. **Simulator:** a drive is a Markov chain on (down, yards to go, field position); a game is 11 drives per team.
   The opponent always plays like real coaches.
3. **Strategies:** each game is a Bernoulli trial (win / loss). Strategies play the same games with the same random
   numbers, so differences are paired; 95% CIs by the CLT.
4. **Search:** each rule alone, then the best combination by successive halving, then opponent strength.
5. **Handbook:** the best strategy's call on every 4th down, next to P(1st down), P(FG good) and what coaches do.

## Assumptions

- No clock or penalties; drives left stand in for the clock.
- 3rd and 4th down share one play distribution (4th-down attempts are rare).
- The opponent does not adapt; opponent strength only changes its offense.
