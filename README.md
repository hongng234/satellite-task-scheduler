<!--
  [Ghi chú tiếng Việt — xoá sau khi đọc]
  - README là "trang chủ" của repo: người khác (nhà tuyển dụng, reviewer) mở GitHub sẽ thấy file này đầu tiên.
  - Cấu trúc: tóm tắt → kết quả → cách chạy → model → kiểm thử/validation → debugging → quá trình phát triển.
  - Phần "Model" giữ lại ký hiệu bạn viết trước đó (x_i, v_i, e_i, m_i, B_s, M_s, K_s, T_i, [a_i, b_i]) và viết nốt phần ràng buộc.
  - Phần "Debugging" và "Validation" chỉ ghi những gì ĐÃ làm thật trong project, để khi phỏng vấn bạn giải thích được.
  - Phần "Development process" nói rõ bạn dùng Claude Code — trung thực, và bạn vẫn phải tự hiểu code.
    Câu "Design decisions ... are my own" là mình viết thay bạn: chỉ giữ nếu đúng, còn không thì sửa lại cho khớp phần bạn thực sự tự làm.
  - Số liệu trong bảng lấy từ lần chạy benchmark thật (time limit 30s, 8 threads). Nếu chạy lại trên máy khác, runtime có thể khác một chút.
-->

# Satellite Task Scheduler

Schedules Earth-observation captures for a satellite constellation over a 48 h horizon.
It compares an exact **OR-Tools CP-SAT** model against two **greedy heuristics** on simulated
datasets from 250 to 66k observation opportunities.

## Results

Benchmark: 30 s CP-SAT time limit, 8 threads. Every schedule passed the independent constraint checker.

| Scenario | Opportunities | CP-SAT | Greedy | Improved greedy | CP-SAT vs greedy |
|---|---|---|---|---|---|
| tiny | 252 | 2 941.52 · **OPTIMAL** (0.04 s) | 2 941.52 | 2 941.52 | 0% |
| small | 1 982 | 14 143.54 · 200/200 requests · **OPTIMAL** (0.48 s) | 14 055.89 · 196 | 14 055.89 · 196 | +0.62% |
| medium | 9 964 | 36 769.41 · 467 requests | 36 579.48 · 456 | 36 579.48 · 456 | +0.52% |
| large | 65 795 | 106 174.31 · 762 requests | 106 164.25 · 761 | 106 075.82 · 761 | +0.01% |
| large, no warm start | 65 795 | 61 552.92 | 106 164.25 | 106 075.82 | **−42%** |

- **Small instances:** CP-SAT proves the optimal schedule in under 0.5 s and serves more requests than greedy.
- **Large instances:** without a warm start, CP-SAT spends its time limit in presolve and search and ends 42% below greedy.
  Warm-starting it from the greedy schedule (a complete solution hint) guarantees CP-SAT ≥ greedy.
- A detailed explanation of the benchmark, in Vietnamese and English, is in [docs/scheduler_comparison.md](docs/scheduler_comparison.md).

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows (use bin/activate on Linux/macOS)
pip install -r requirements.txt

python -m src.generate_dataset                # data/processed/*.parquet (seed 42, reproducible)
python -m src.preview_dataset --rows 200      # inspect the first/last rows of each table
python -m src.run_experiment                  # 3 algorithms -> output/
python -m src.benchmark --generate-only       # only the 4 benchmark datasets -> data/benchmark/<scenario>/
python -m src.benchmark                       # tiny/small/medium/large -> output/benchmark_results.csv
python -m pytest                              # 35 tests
```

`make dataset | datasets | run | preview | benchmark | test` run the same commands (`make help` lists all).

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
docs/                   benchmark explanation (VI + EN)
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
orbit geometry is simulated. In CP-SAT, constraint 3 is an `AddNoOverlap` over **optional** intervals of
length `T_i + δ`. All numbers are scaled to integers (value ×100, energy ×10, data ×100), with conservative
rounding (consumption rounded up, capacity rounded down).

## Testing

`python -m pytest` runs 35 tests in about 1 s:

| File | What it covers |
|---|---|
| `tests/test_generate_dataset.py` | Reproducibility (same seed → identical tables), deterministic ids, windows inside the request range, orbit-period spacing, empty-data edge case |
| `tests/test_greedy.py` | Unit tests of the start-time rule of both greedy variants; battery, one-capture-per-request and input validation |
| `tests/test_cp_sat.py` | **Regression tests** for fixed bugs (non-optional intervals → infeasible, too-short window → `KeyError`, ignored slew buffer); optimality on a small instance; CP-SAT ≥ greedy with and without the hint |
| `tests/test_validation.py` | End-to-end: all 3 schedulers produce valid schedules; the constraint checker detects injected violations; Pydantic model validation |

## Validation

- **Independent constraint checker:** `src/benchmark.py::check_schedule` re-checks every schedule against
  all constraints, using the original float data rather than the solver's integer model. The benchmark
  reports it in the `valid` column.
- **Reproducible data:** fixed `--start_time`, seeded RNGs and sequential ids. Two runs with the same seed
  produce byte-identical Parquet files.
- **Cross-check between algorithms:** on instances solved to `OPTIMAL`, CP-SAT must be ≥ both greedy
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

## Development process

Developed with **Claude Code** (an AI coding agent) as an assistant for planning changes, reviewing logic, suggested hint logic, debugging, writting tests, and the solver. Design decisions, model choices, writing the base codes, and the final review of the results are my own.
