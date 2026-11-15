
# Satellite Task Scheduler

Schedules Earth-observation captures for a satellite constellation over a 48 h horizon.
It compares an exact OR-Tools CP-SAT model against two greedy heuristics on simulated
datasets from 250 to 66k observation opportunities.

## Results

Benchmark: 30 s CP-SAT time limit, 8 threads. Every schedule passed the independent constraint checker.

| Scenario | Opportunities | CP-SAT | Greedy | Improved greedy | CP-SAT vs greedy |
|---|---|---|---|---|---|
| tiny | 252 | 2 941.52 · OPTIMAL (0.04 s) | 2 941.52 | 2 941.52 | 0% |
| small | 1 982 | 14 143.54 · 200/200 requests · OPTIMAL (0.48 s) | 14 055.89 · 196 | 14 055.89 · 196 | +0.62% |
| medium | 9 964 | 36 769.41 · 467 requests | 36 579.48 · 456 | 36 579.48 · 456 | +0.52% |
| large | 65 795 | 106 174.31 · 762 requests | 106 164.25 · 761 | 106 075.82 · 761 | +0.01% |
| large, no warm start | 65 795 | 61 552.92 | 106 164.25 | 106 075.82 | −42% |

- On small instances, CP-SAT proves the optimal schedule in under 0.5 s and serves more requests than greedy.
- On large instances without a warm start, CP-SAT uses up its time limit in presolve and search and ends 42% below greedy.
  Warm-started from the greedy schedule (a complete solution hint), it never ends below greedy.
- The medium and large CP-SAT rows stop at the time limit (`FEASIBLE`), so they vary a little between runs.
  Later reruns gave 36 766.02 to 36 777.93 on medium, and 61 478.11 to 62 082.39 for large without warm start.

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows (use bin/activate on Linux/macOS)
pip install -r requirements.txt

python -m src.generate_dataset                # data/processed/*.parquet (seed 42, reproducible)
python -m src.preview_dataset --rows 200      # inspect the first/last rows of each table
python -m src.run_experiment                  # 3 algorithms -> output/
python -m src.benchmark --generate-only       # only the 4 benchmark datasets -> data/benchmark/<scenario>/
python -m src.benchmark                       # tiny/small/medium/large -> output/benchmark_results.csv
python -m pytest                              # 62 tests
```

`make dataset | datasets | run | preview | benchmark | test` run the same commands (`make help` lists all).
`make lint` and `make typecheck` run ruff and mypy, with their settings in `pyproject.toml`.

## Project structure

```
src/
  generate_dataset.py   simulated satellites, requests, observation windows (reproducible)
  models.py             Pydantic models; ScheduleResult is the common output of every scheduler
  scheduler_cp_sat.py   exact CP-SAT model + greedy warm start
  scheduler_greedy.py   GreedyScheduler (baseline) and ImprovedGreedyScheduler
  run_experiment.py     run the 3 algorithms on one dataset
  benchmark.py          multi-size benchmark + independent constraint checker
  preview_dataset.py    head/tail viewer for the Parquet files
tests/                  pytest suite (unit, regression and end-to-end tests)
```

## Model

Notation:

- `i`: opportunity; `r(i)`: its request; `s(i)`: its satellite
- `x_i ∈ {0,1}`: 1 if opportunity `i` is selected
- `v_i`: request value; `e_i`: energy (Wh); `m_i`: data volume (GB); `T_i`: capture duration (s)
- `[a_i, b_i]`: observation window; `t_i`: chosen start time
- `B_s`, `M_s`: battery and memory capacity of satellite `s`; `K_s`: max captures per orbit
- `δ`: slew buffer (default 30 s)

Objective:

```
max  Σ_i v_i · x_i
```

Constraints:

```
1. One capture per request         Σ_{i : r(i)=r} x_i ≤ 1                       for every request r
2. Inside the window               a_i ≤ t_i   and   t_i + T_i ≤ b_i             if x_i = 1
3. No overlap + slew buffer        t_j ≥ t_i + T_i + δ  or  t_i ≥ t_j + T_j + δ   for selected i, j on one satellite
4. Battery                         Σ_{i : s(i)=s} e_i · x_i ≤ B_s
5. Memory                          Σ_{i : s(i)=s} m_i · x_i ≤ M_s
6. Captures per orbit              Σ_{i in orbit block k of s} x_i ≤ K_s,   k = floor((a_i − t_0) / orbit_period_s)
```

Simplifications: battery and memory are budgets for the whole horizon (no recharge, no downlink), and the
orbit geometry is simulated. In CP-SAT, constraint 3 is an `AddNoOverlap` over optional intervals of
length `T_i + δ`. All numbers are scaled to integers (value ×100, energy ×10, data ×100). Consumption is
rounded up and capacity rounded down, so the integer model never allows more than the real data.

Because durations are rounded up to whole seconds, a CP-SAT capture can be slightly longer than the request
needs (an 8.2 s capture is reported as 9 s). The greedy schedulers report the exact float duration.

Before solving, every scheduler checks its input tables (`validate_tables` in `src/models.py`): ids must exist,
the slew buffer must be ≥ 0, durations must be > 0, and an opportunity's duration must equal its request's.

## Testing

`python -m pytest` runs 62 tests in about 2 s:

| File | What it covers |
|---|---|
| `tests/test_generate_dataset.py` | Reproducibility (same seed → identical tables), deterministic ids, windows inside the request range, orbit-period spacing, empty-data edge case |
| `tests/test_greedy.py` | Unit tests of the start-time rule of both greedy variants; battery, one-capture-per-request and input validation |
| `tests/test_cp_sat.py` | Regression tests for fixed bugs (non-optional intervals → infeasible, too-short window → `KeyError`, ignored slew buffer, orbit block rounding); optimality on a small instance; CP-SAT ≥ greedy with and without the hint |
| `tests/test_validation.py` | End-to-end: all 3 schedulers produce valid schedules; the constraint checker detects injected mistakes (duplicate request, capture outside its window or too short, wrong ids, value or priority, wrong total, infeasible opportunity, unknown id); input checks of every scheduler; Pydantic model validation |

## Validation

- Independent constraint checker: `src/benchmark.py::check_schedule` checks every schedule against
  the constraints, using the original float data and not the solver's integer model. It also checks that
  each capture matches its opportunity and request (ids, duration, value, priority) and that the reported
  total is the sum of the values. The benchmark reports it in the `valid` column.
- Reproducible data: fixed `--start_time`, seeded RNGs and sequential ids. Two runs with the same seed
  produce byte-identical Parquet files.
- Cross-check between algorithms: on instances solved to `OPTIMAL`, CP-SAT must be ≥ both greedy
  variants. The test suite asserts this.

## Debugging notes

Issues found and how they were diagnosed:

| Symptom | Diagnosis | Fix |
|---|---|---|
| CP-SAT infeasible or wrong on normal data | Intervals were mandatory, so `NoOverlap` also applied to unselected opportunities | `new_optional_interval_var(..., is_present=x_i)` |
| Crash on some datasets | Opportunities dropped from the model (window shorter than the capture) were still indexed later | Skip ids that have no variable |
| CP-SAT returned objective 0, or 42% below greedy, on 66k opportunities | CP-SAT search log (`log_search_progress`): about 15 s of presolve, then no good solution before the time limit | Greedy warm start via `model.add_hint` |
| Hint ignored | The log said `solution hint is incomplete: 196688 out of 295032`. Fixing variables to the hint (`fix_variables_to_their_hinted_value`) proved the hint itself was feasible | Hint every variable (`x`, `start`, `end`) |
| Hint slightly infeasible | Greedy uses float durations, CP-SAT uses rounded-up integers | Re-place greedy captures on the integer grid before hinting |
| CP-SAT and the checker could disagree on the orbit block | CP-SAT rounded the time offset down to whole seconds before dividing by a non-integer orbit period; a script comparing both formulas found 19 of 9 964 opportunities in a different block on medium | CP-SAT reuses the greedy `_orbit_block` function |
| Checker accepted wrong schedules | Hand-made schedules with a 0 s capture, a wrong request id or an inflated value were reported as valid, and an unknown opportunity id crashed it | More checks in `check_schedule`, one test per injected mistake |

## Development process

The project idea and the backbone code (dataset generator, Pydantic models, CP-SAT model and baseline greedy) are my own.
I then used Claude Code (an AI coding agent) to fix the code, write tests, audit the schedulers and update the project.

When CP-SAT struggled on large sets of opportunities (objective 0, or 42% below greedy), Claude suggested warm-starting it
with a solution hint built from the greedy schedule. I decided to adopt it and confirmed it with the benchmark:
on the large scenario, CP-SAT went from 61 552.92 without the hint to 106 174.31 with it (see [Debugging notes](#debugging-notes)).

I reviewed every change, and every benchmark schedule is verified by an independent constraint checker (`check_schedule`).
