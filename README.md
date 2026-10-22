# 4th down: go for it, kick, or punt?

Applied Probability (T-811, Reykjavík University), project 1.
Eiríkur Bernharðsson, Davíð Rúnar Matthíasson, Grétar Másson.

Which rule for when to go for it on 4th down wins the most games? We estimate probabilities from NFL play-by-play data
(2018–2025, nflverse), simulate whole games as a Markov chain, and compare strategies with Monte Carlo.

**Start with [`results.ipynb`](results.ipynb)**: all plots, tables, the handbook and the conclusions, read from saved results.

## Main result

- **Best single rule:** "go for it with 5 yards or less to go" wins about 53% against real coaches, about +3 pp.
- **Best combination:**
  - go on ≤ 5 yards, but never inside our own 20
  - go on ≤ 6 yards in the dead zone, 35–50 yards from the end zone
  - go when trailing with ≤ 2 drives left
  - kick a field goal late when it changes the number of scores
  - when leading late, never go for it in our own half
  - otherwise kick a field goal up to 57 yards, else punt
- **The combination wins 54.2%:** +4.0 pp [3.4, 4.6] over the coaches, and +1.3 pp [1.0, 1.6] over the best single rule.
- **Never going for it and always going for it both lose,** and so does going for it at random. *Where* you go for it matters, not how often.
- **The stronger the opponent, the more aggression pays:** +3.6 pp against a weak team, +4.6 pp against a strong team, each with its own best strategy.

## Files

| File | What it is |
|---|---|
| `results.ipynb` | Shows everything: plots, tables, handbook, conclusions. Reads the saved results; runs in seconds. |
| `compute.ipynb` | Works through the data and runs all simulations for an equal opponent; saves `results.json` (~45 min). |
| `opponents_search.py` | Extension: the same search against a weak and a strong opponent; saves `results_opponents.json` (~1 h). |
| `model.py` | Data, probabilities, the simulator (Markov chain), strategies, successive halving, handbook. |
| `plots.py` | The plots used in `results.ipynb`. |
| `deep_tradeoff.py` | Go for it vs punt deep in our own half, in net points (`deep_tradeoff.csv`). |
| `results.json`, `results_opponents.json`, `results_opp_*.json` | Saved simulation results. |
| `presentation/` | The slides (`.pptx` with speaker notes, and a PDF). |
| `requirements.txt` | Python packages. |

## Run

```bash
pip install -r requirements.txt
jupyter notebook results.ipynb
```

The play-by-play data (about 200 MB) is not included. `model.load_data()` downloads it from nflverse into `data/` on
first use (`data/pbp_<year>.csv.gz`) and caches it in `data/pbp.pkl`.

To redo the simulations, run `compute.ipynb` and then `opponents_search.py`. Every step uses fixed random seeds, so a
rerun gives the same numbers. Setting `M.SPEED = 0.2` at the top of `compute.ipynb` gives a quick, noisier test run.

## Method in short

1. **Probabilities from data:**
   - yards gained per play, sampled from real plays with the same down and yards to go (3rd and 4th down pooled)
   - P(1st down | yards to go)
   - P(field goal good | distance), by logistic regression
   - punts, drawn from real punts kicked within ±5 yards of the same spot
   - what real coaches do on 4th down: P(go | yards to go, field position)
2. **Simulator:** a drive is a Markov chain on (down, yards to go, field position). A game is 11 drives per team, and the number of drives left replaces the clock. The opponent always plays like the real coaches.
3. **Monte Carlo:** each game is a Bernoulli-type trial (win 1, tie ½, loss 0). All strategies play the same games with the same random numbers, so differences are paired. The 95% CIs come from the CLT. Coach vs coach gives 49.9% (p = 0.60), so the simulator has no built-in bias.
4. **Step 1:** each rule on its own.
   - the field-goal range (57 yards)
   - aggressiveness (random vs short yardage)
   - a heatmap of "go if ≤ X yards and at most Y yards from the end zone", on 100,000 paired games per cell
   - a dead-zone heatmap
   - the single strategies
5. **Step 2:** combinations.
   - The candidates are the top 5 of each heatmap plus the goal-line and late-game rules: 4,224 candidates.
   - They are searched by successive halving, 1,000 → 64,000 games, keeping the best quarter each round.
   - The top 5 are then re-tested on 50,000 fresh paired games (against the winner's curse).
6. **Extension:** the same search against a weak and a strong opponent, whose plays are drawn only from the bottom or top third of teams by EPA per play.

## Assumptions and choices

- **No clock or penalties;** the number of drives left stands in for the clock. No 2-point tries or overtime.
- **4th downs use 3rd-down plays,** because real 4th-down attempts are rare and biased.
- **The opponent doesn't adapt to our strategy.** Opponent strength only changes its offense.
- **The simulated coaches use their quarter 1–3 behaviour all game,** so they don't react to the score late. As a check, we re-ran the main analysis with score-aware coaches, using their real 4th-quarter decisions per score state. The gain fell from 4.0 to 3.6 pp, and the handbook barely changed.
- **Punts:** the kick distance and return yards of the sampled real punt are both added to the punt. Punt-return touchdowns and muffed punts are included.
- **Dead zone:** the five best ranges go into the search. 25–55 (+0.61 pp) was replaced by the next range, 40–55 (+0.57 pp); the two are within noise.
- **Comparisons only:** the results compare strategies; they aren't exact win probabilities.
