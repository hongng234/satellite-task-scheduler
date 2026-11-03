
### 1. Problem

A constellation of satellites must serve imaging **requests**. Each satellite–request pair may have a few **opportunities**: time windows in which that satellite can image the request's area.

The goal is to pick opportunities that **maximize total value** while satisfying 5 constraints:

| # | Constraint | Meaning |
|---|---|---|
| 1 | Each request at most once | No duplicate captures |
| 2 | No overlap + slew buffer | ≥ 30 s idle between two captures on one satellite (re-orientation) |
| 3 | Battery | Total capture energy ≤ battery capacity |
| 4 | Memory | Total data volume ≤ onboard memory |
| 5 | Per-orbit cap | Captures per orbit ≤ `max_observations_per_orbit` |

Objective (`x_i = 1` if opportunity i is selected, otherwise 0):

```
maximize   Σ value_i × x_i
```

### 2. The three algorithms

**Greedy (baseline)**, in `scheduler_greedy.py`: rank opportunities by priority, then by `value / duration`. Accept each one in order if it breaks no constraint. The capture always starts at `window_start`; if the satellite is busy then, the opportunity is skipped.

**Improved greedy** (`ImprovedGreedyScheduler`): same as the baseline, but it shifts the start to the **earliest free time** inside the window.

**CP-SAT**, in `scheduler_cp_sat.py`: an exact constraint-programming model solved by Google OR-Tools.
- It can **prove optimality** (`OPTIMAL`).
- If the time limit is reached, it returns the best schedule found so far (`FEASIBLE`).
- **Warm start (hint):** CP-SAT starts from the better of the two greedy schedules, so its result is never worse than greedy.

### 3. How the benchmark numbers are produced

For each dataset size, `src/benchmark.py` runs these steps:

```
1. Generate the dataset (fixed seed → reproducible)
2. Run all 3 algorithms on the SAME dataset
3. Re-check every schedule against all 5 constraints → "valid" column
4. Compute the metrics → output/benchmark_results.csv
```

Metrics:

| Column | Formula | Meaning |
|---|---|---|
| `objective_value` | `Σ value` of the selected captures | Total value served |
| `scheduled_requests` | number of captures | Requests served |
| `served_ratio` | `scheduled_requests / total requests` | Share of requests served |
| `gain_vs_greedy_pct` | `(objective / objective_greedy − 1) × 100` | Gain over the baseline greedy |
| `runtime_seconds` | wall-clock time of `solve()` | CP-SAT is capped by `--time-limit` |
| `status` | `OPTIMAL` / `FEASIBLE` | `OPTIMAL` = proven best |
| `valid` | `True` if no constraint is violated | Independent correctness check |

### 4. Results (30 s time limit, 8 threads)

| Scenario | Opportunities | CP-SAT | Greedy | Improved greedy | CP-SAT gain |
|---|---|---|---|---|---|
| tiny | 252 | 2 941.52 · **OPTIMAL** in 0.04 s | 2 941.52 | 2 941.52 | 0% |
| small | 1 982 | 14 143.54 · 200/200 requests · **OPTIMAL** in 0.48 s | 14 055.89 · 196 | 14 055.89 · 196 | +0.62% |
| medium | 9 964 | 36 769.41 · 467 requests | 36 579.48 · 456 | 36 579.48 · 456 | +0.52% |
| large | 65 795 | 106 174.31 · 762 requests | 106 164.25 · 761 | 106 075.82 · 761 | +0.01% |
| large, **no hint** | 65 795 | 61 552.92 | 106 164.25 | 106 075.82 | **−42%** |

Every schedule passed the constraint check (`valid = True`).

### 5. Takeaways

- **Small instances:** CP-SAT finds and proves the optimal schedule in under 0.5 s. Greedy is close but misses a few requests.
- **Large instances:** greedy runs in under 0.5 s, while CP-SAT spends about 15 s in presolve alone. Without a warm start, CP-SAT ends 42% below greedy. With the greedy hint it is always ≥ greedy; a longer time limit is needed for larger gains.
- **Improved greedy ≈ baseline:** windows are only about 3× the capture duration, shorter than the 30 s slew buffer, so there is rarely room to shift a capture.
- **Model note:** battery and memory are budgets for the whole 48 h horizon (no recharge, no downlink). On large instances they are the binding constraints: each satellite only has memory for about 40 captures, so only 762/2000 requests (38%) can be served, whatever the algorithm.
