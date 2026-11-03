
# Satellite Task Scheduler

Schedules Earth-observation captures for a satellite constellation over a planning horizon (default 48 h).
It compares an exact optimizer (OR-Tools CP-SAT) with two greedy heuristics. All data is **simulated** by
`src/generate_dataset.py`; this is not a real orbital model. The README is the human-facing overview,
and `docs/scheduler_comparison.md` explains the benchmark in Vietnamese and English.

## Commands
Use the venv interpreter `.venv/Scripts/python.exe` (Windows, Python 3.14). Run everything from the repo root
with `python -m src.<module>`, because imports are `from src.x import ...`.

```bash
python -m pytest                               # 35 tests, ~1 s (make test)
python -m src.generate_dataset                 # data/processed/*.parquet (make dataset)
python -m src.run_experiment                   # 3 algorithms -> output/ (make run); --no-hint disables warm start
python -m src.preview_dataset --rows 200       # head/tail of each table (make preview)
python -m src.benchmark                        # tiny/small/medium/large -> output/benchmark_results.csv (make benchmark)
python -m src.benchmark --scenarios xlarge --time-limit 120
python -m src.benchmark --generate-only        # only write data/benchmark/<scenario>/*.parquet (make datasets)
```

## Layout
- `src/models.py`: Pydantic models. `ScheduleResult` / `ScheduledObservation` are the common output of every scheduler.
- `src/generate_dataset.py`: `generate_all(...)` builds the 3 DataFrames in memory, and `save_dataset(...)` writes Parquet.
  The generator is reproducible (fixed `--start_time`, seeded RNGs, ids `OPP-000001`). Satellite passes are
  anchored to the common `start_time` plus a per-AOI orbit fraction.
- `src/scheduler_cp_sat.py`: `SatelliteTaskingScheduler` (CP-SAT model and greedy warm start).
- `src/scheduler_greedy.py`: `GreedyScheduler` (baseline: start at `window_start` or skip) and
  `ImprovedGreedyScheduler` (earliest free start in the window). They differ only in `_find_start`.
  Both have `from_dataframes(...)`.
- `src/run_experiment.py`: `run_all(...)` runs the 3 algorithms and writes the outputs.
- `src/benchmark.py`: multi-size scenarios. `check_schedule(...)` is the independent constraint checker.
- `src/preview_dataset.py`: head/tail viewer.
- `tests/`: pytest suite. `conftest.py` provides `small_dataset` (session fixture from the real generator) and
  `write_dataset` plus the `make_satellite` / `make_request` / `make_opportunity` helpers for hand-made cases.
  `pytest.ini` sets `pythonpath = .`.

## Model (shared by all schedulers)
Decision `x_i ∈ {0,1}` per opportunity. Maximize `Σ value_i · x_i` subject to:
1. At most one opportunity per request.
2. The capture lies inside its window: `window_start ≤ start` and `start + duration ≤ window_end`.
   The window lies inside the request's `[earliest_start, latest_end]`.
3. No overlap per satellite, with a slew buffer: `next_start ≥ prev_end + slew_buffer` (default 30 s).
4. Battery and 5. memory per satellite, as budgets for the whole horizon (no recharge, no downlink).
6. Per orbit block `floor((window_start − base_time) / orbit_seconds)`: at most `max_observations_per_orbit` captures.

## CP-SAT pitfalls (already handled, covered by tests — do not regress)
- Intervals **must be optional** (`new_optional_interval_var(..., is_present=x)`); otherwise the model is infeasible.
- Opportunities dropped from the model (window shorter than the capture) have no variable. Every later loop must skip them.
- CP-SAT is integer-only. Times are seconds since `base_time`: ceil for starts and durations, floor for window ends.
  Scaling: value ×100, energy ×10, data ×100; consumption is rounded up and capacity rounded down.
- Build sums with `cp_model.LinearExpr.weighted_sum / sum`, not Python `sum()`.
- **Warm start:** the better of the two greedy schedules is re-placed on the integer grid and hinted for
  **every** variable (`x`, `start`, `end`). An incomplete hint is ignored on large models.
- Only read `solver.value(...)` when the status is OPTIMAL or FEASIBLE.
- On ≥ 50k opportunities presolve takes ~15 s, so use a larger `--time-limit` to see improvement beyond greedy.

## Debugging tips
- CP-SAT log: set `solver.parameters.log_search_progress = True`. Look for `solution hint is complete and is feasible`,
  and for the `#1 ... best:` line to see which worker found the first solution.
- To check whether a hint is feasible: `solver.parameters.fix_variables_to_their_hinted_value = True`.
- To check any schedule: `src.benchmark.check_schedule(result, satellites, requests, opportunities, slew_buffer)`
  returns a list of violations (empty = valid).

## Conventions
- Code comments are in **English**, explain each line or block, and give formulas with `Formula:` / `Ex:`
  blocks (see `scheduler_cp_sat.py`).
- The user writes in Vietnamese; explanations in chat are in Vietnamese. Notes meant only for the user may be
  left as Vietnamese comments marked "xoá sau khi đọc"; the user removes them.
- Every scheduler returns `ScheduleResult`. A new algorithm should follow that, be added to `run_all` in
  `run_experiment.py`, and get a case in `tests/test_validation.py`.
- After changing code: run `python -m pytest`. After changing a scheduler, also run
  `python -m src.benchmark --scenarios tiny small medium` and check that `valid` is True for every row.
- When fixing a bug, add a regression test, using a hand-made dataset from `conftest.py` where possible.
- Do not claim results or tests that do not exist. README numbers come from real benchmark runs, so update
  them when behaviour changes.
- `data/`, `output/`, `*.parquet`, `*.csv`, `*.json` are git-ignored.
