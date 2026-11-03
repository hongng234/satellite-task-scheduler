# So sánh 3 thuật toán Scheduler / Comparing the 3 Schedulers

- [Tiếng Việt](#tiếng-việt)
- [English](#english)

---

## Tiếng Việt

### 1. Bài toán

Có nhiều vệ tinh và nhiều **request** (yêu cầu chụp ảnh một vùng AOI). Mỗi cặp vệ tinh–request có thể có vài **opportunity**, tức là những khung thời gian (window) mà vệ tinh đó bay qua và chụp được vùng đó.

Mục tiêu là chọn một tập opportunity sao cho **tổng value lớn nhất**, đồng thời thoả 5 ràng buộc:

| # | Ràng buộc | Ý nghĩa |
|---|---|---|
| 1 | Mỗi request tối đa 1 lần | Không chụp cùng một yêu cầu 2 lần |
| 2 | Không chồng lấn + slew buffer | Giữa hai lần chụp trên cùng vệ tinh phải nghỉ ≥ 30s để xoay vệ tinh |
| 3 | Pin | Tổng năng lượng các lần chụp ≤ dung lượng pin |
| 4 | Bộ nhớ | Tổng dữ liệu các lần chụp ≤ dung lượng bộ nhớ |
| 5 | Giới hạn mỗi vòng quỹ đạo | Số lần chụp trong 1 vòng ≤ `max_observations_per_orbit` |

Công thức mục tiêu (`x_i = 1` nếu chọn opportunity i, ngược lại `x_i = 0`):

```
maximize   Σ value_i × x_i
```

### 2. Ba thuật toán

**Greedy (baseline)**, trong `scheduler_greedy.py`:
1. Xếp hạng opportunity: priority cao trước, sau đó đến `value / duration` cao trước.
2. Duyệt lần lượt từ trên xuống; opportunity nào không vi phạm ràng buộc thì nhận.
3. Luôn chụp ở đúng `window_start`. Nếu lúc đó vệ tinh đang bận thì bỏ qua.

**Greedy nâng cấp** (`ImprovedGreedyScheduler`): giống greedy, nhưng nếu `window_start` bị bận thì dời giờ chụp tới **thời điểm trống sớm nhất** vẫn nằm trong window.

**CP-SAT**, trong `scheduler_cp_sat.py`: viết bài toán thành một mô hình toán học rồi giao cho solver OR-Tools CP-SAT tìm lời giải tốt nhất.
- Solver có thể **chứng minh** lời giải là tối ưu (trạng thái `OPTIMAL`).
- Nếu hết thời gian (`--time-limit`), solver trả về lời giải tốt nhất đã tìm được (`FEASIBLE`).
- **Warm start (hint):** CP-SAT bắt đầu từ lịch tốt hơn trong 2 lịch greedy, nên kết quả luôn ≥ greedy. Không có hint thì trên dataset lớn CP-SAT hết giờ khi vẫn còn kém greedy rất xa.

### 3. Benchmark tính ra kết quả như thế nào

Script `src/benchmark.py` làm 4 bước cho mỗi kích thước dataset:

```
1. Sinh dataset (seed cố định → chạy lại ra đúng dữ liệu cũ)
2. Chạy 3 thuật toán trên CÙNG dataset đó
3. Kiểm tra lại từng lịch theo cả 5 ràng buộc → cột "valid"
4. Tính các chỉ số và ghi vào output/benchmark_results.csv
```

Kích thước các scenario:

| Scenario | Vệ tinh | Request | Opportunity (xấp xỉ) |
|---|---|---|---|
| tiny | 3 | 50 | 252 |
| small | 6 | 200 | 1 982 |
| medium | 12 | 500 | 9 964 |
| large | 20 | 2 000 | 65 795 |

Số opportunity xấp xỉ bằng `request × vệ tinh × 0.55 × ~2.5`: 55% là xác suất một vệ tinh nhìn thấy được vùng AOI, và mỗi cặp vệ tinh–request có tối đa 3 window.

**Các chỉ số:**

| Cột | Công thức | Ý nghĩa |
|---|---|---|
| `objective_value` | `Σ value` của các lần chụp được chọn | Tổng giá trị phục vụ được (càng cao càng tốt) |
| `scheduled_requests` | số lần chụp trong lịch | Số request được phục vụ |
| `served_ratio` | `scheduled_requests / tổng request` | Tỉ lệ request được phục vụ |
| `gain_vs_greedy_pct` | `(objective / objective_greedy − 1) × 100` | Hơn greedy baseline bao nhiêu % |
| `runtime_seconds` | thời gian chạy `solve()` | CP-SAT bị giới hạn bởi `--time-limit` |
| `status` | `OPTIMAL` / `FEASIBLE` | `OPTIMAL` = đã chứng minh không có lịch nào tốt hơn |
| `valid` | `True` nếu không vi phạm ràng buộc nào | Lịch nhanh mà sai thì vô giá trị |

Ví dụ với scenario medium: `36 769.41 / 36 579.48 − 1 = 0.0052`, tức **+0.52%**.

### 4. Kết quả (time limit 30s, 8 threads)

| Scenario | CP-SAT | Greedy | Greedy nâng cấp | CP-SAT hơn greedy |
|---|---|---|---|---|
| tiny | 2 941.52 · 45 req · **OPTIMAL** 0.04s | 2 941.52 · 45 | 2 941.52 · 45 | 0% |
| small | 14 143.54 · **200/200 req** · **OPTIMAL** 0.48s | 14 055.89 · 196 | 14 055.89 · 196 | +0.62% |
| medium | 36 769.41 · 467 req · 31.5s | 36 579.48 · 456 | 36 579.48 · 456 | +0.52% |
| large | 106 174.31 · 762 req · 36.5s | 106 164.25 · 761 | 106 075.82 · 761 | +0.01% |
| large, **không hint** | 61 552.92 · 756 req | 106 164.25 | 106 075.82 | **−42%** |

Tất cả lịch đều `valid = True`.

### 5. Đọc kết quả thế nào

- **Dataset nhỏ:** CP-SAT tìm được lời giải tối ưu và chứng minh được điều đó trong chưa tới 0.5s. Greedy đã gần tối ưu nhưng vẫn bỏ sót một vài request.
- **Dataset lớn:** greedy chạy dưới 0.5s, còn CP-SAT tốn khoảng 15s chỉ để presolve. Nếu không có hint, CP-SAT kém greedy 42%. Có hint thì CP-SAT luôn ≥ greedy; muốn cải thiện thêm cần tăng `--time-limit`.
- **Greedy nâng cấp gần như bằng baseline:** window chỉ dài khoảng 3 lần thời gian chụp, ngắn hơn khoảng nghỉ 30s, nên hiếm khi dời giờ chụp được.
- **Lưu ý về model:** pin và bộ nhớ tính cho cả 48h, không sạc lại và không tải dữ liệu xuống, nên đây là ràng buộc chặt nhất khi dataset lớn. Ví dụ ở scenario large, mỗi vệ tinh chỉ đủ bộ nhớ cho khoảng 40 lần chụp, nên chỉ phục vụ được 762/2000 request (38%), dù thuật toán nào cũng vậy.

### 6. Chạy lại

```bash
python -m src.benchmark                                  # tiny, small, medium, large
python -m src.benchmark --scenarios large --no-hint      # xem CP-SAT khi không có hint
python -m src.benchmark --scenarios large --time-limit 120
```

---

## English

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
