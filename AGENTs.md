# Grid Transit Simulation v1
用 Python 3.10+ 在 Linux 环境实现，代码严格只有三个文件：demand_generation.py、mode_set.py、main.py。
依赖固定为：numpy、pandas、networkx，外加少量标准库。
**Summary**
- Build a stdlib-only Python framework with exactly three code files: `demand_generation.py`, `mode_set.py`, and `main.py`.
- Use a fixed `9x9` Manhattan grid, a `180`-minute service horizon, `7` vehicles, capacity `30`, and a central hub at `(4, 4)`.
- Evaluate a built-in scenario grid `lambda ∈ {20, 40, 60}`, `hs ∈ {0.0, 0.5, 1.0}`, `ht ∈ {0.0, 0.5, 1.0}` for `27` scenarios total, with one random demand sample per scenario.
- Define `net_expenditure` as total active vehicle travel time. Mode 1 sets the per-scenario benchmark; Modes 2-4 are only comparable if they serve every request and keep `net_expenditure <= Mode1 net_expenditure`.

**Public Interfaces**
- `demand_generation.py`
  - `TripRequest` dataclass with `request_id`, `origin`, `destination`, `departure_time`.
  - `generate_requests(lambda_value, hs, ht, seed, grid_size=9, horizon=180) -> list[TripRequest]`.
- `mode_set.py`
  - `manhattan_distance(a, b) -> int`.
  - `evaluate_mode_1(requests, scenario) -> dict`
  - `evaluate_mode_2(requests, scenario, benchmark_expenditure) -> dict`
  - `evaluate_mode_3(requests, scenario, benchmark_expenditure) -> dict`
  - `evaluate_mode_4(requests, scenario, benchmark_expenditure) -> dict`
  - All four mode functions return the exact same flat result schema:
    `scenario_id, lambda, hs, ht, seed, mode_id, mode_name, feasible, feasibility_reason, total_requests, served_requests, unserved_requests, benchmark_expenditure, net_expenditure, total_wait, total_walk, total_onboard, total_service_time, avg_wait, avg_walk, avg_onboard, avg_service_time`.
- `main.py`
  - Builds the scenario grid, generates one seeded demand sample per scenario, runs Mode 1 first, then Modes 2-4, and exports `scenario_results.csv` and `scenario_results.json` in the workspace root.

**Implementation Changes**
- Demand generation:
  - Treat `lambda` as the Poisson mean number of requests over the full horizon.
  - Keep `hs` and `ht` normalized to `[0, 1]`.
  - Sample origins and destinations from fixed hotspot-weight maps blended with uniform demand: low `hs` is nearly uniform, high `hs` concentrates origins near `(1,1)` and destinations near `(7,7)`.
  - Sample departure times from a uniform-to-peaked mixture: low `ht` is uniform, high `ht` concentrates around two deterministic peaks at minutes `45` and `120`.
  - Resample destination if it equals origin. Use `seed = BASE_SEED + scenario_index` so each scenario is reproducible and all modes see the exact same requests.
- Mode 1: fixed route
  - Use an 8-stop clockwise loop: `(1,1) -> (1,4) -> (1,7) -> (4,7) -> (7,7) -> (7,4) -> (7,1) -> (4,1) -> back`.
  - Start 7 vehicles evenly phased on that loop at time `0`.
  - Each request walks to the best boarding stop and from the best alighting stop. Assign it to the earliest feasible vehicle pass with spare capacity on every traversed arc.
- Mode 2: deviated route
  - Reuse the Mode 1 loop and initial phasing.
  - Allow optional stops that are exactly one Manhattan step off any loop segment. A detour leaves the loop and rejoins the same segment, adding a fixed `2` time units.
  - For each request, choose the mandatory-stop or optional-stop combination that minimizes that rider’s local service time, subject to capacity and cumulative operator time not exceeding the Mode 1 benchmark.
- Mode 3: DRT rolling horizon
  - Start all 7 vehicles idle at the hub `(4,4)`.
  - Replan every `10` minutes with a `20`-minute lookahead window.
  - Insert unscheduled requests in departure-time order using a rolling-horizon pickup/dropoff insertion heuristic over current vehicle plans.
  - Choose the feasible insertion with the lexicographically best score: lowest request wait, then lowest onboard time, then lowest added vehicle travel time, then lowest vehicle id. `T_walk = 0` for all requests.
  - If any request cannot be inserted within capacity or if total operator time would exceed the benchmark, mark the mode infeasible.
- Mode 4: hub-and-spoke
  - Use four fixed spokes from `(4,4)` to the north, east, south, and west edges.
  - Vehicles shuttle hub-to-edge-to-hub and are dispatched from the hub in deterministic round-robin spoke order.
  - Each request walks to the nearest spoke stop, rides inbound to the hub, transfers if needed, rides outbound, then walks to destination.
  - Count transfer waiting inside `T_wait`, not `T_onboard`.
- Shared rules:
  - One grid edge equals one time unit for both walking and riding.
  - A mode is infeasible if any request is unserved or capacity is exceeded anywhere.
  - If Mode 1 is infeasible for a scenario, export Mode 1’s infeasible row and mark Modes 2-4 infeasible with `feasibility_reason="benchmark_mode_infeasible"`.

**Test Plan**
- Re-running `main.py` with the same base seed reproduces identical requests and identical CSV/JSON outputs.
- Every scenario produces exactly 4 exported result rows and the JSON is the same data as the CSV in list-of-dicts form.
- Mode 3 always reports `total_walk == 0` and Mode 4 includes transfer delay in `total_wait`.
- For each feasible scenario, Modes 2-4 satisfy `served_requests == total_requests` and `net_expenditure <= benchmark_expenditure`.
- At least one forced-overload scenario check is included in code comments or a simple manual smoke path to confirm infeasibility handling when capacity or benchmark limits are violated.

**Assumptions**
- No external packages; use only the Python standard library (`dataclasses`, `math`, `random`, `csv`, `json`).
- v1 is a deterministic heuristic simulator, not a global optimizer.
- No Monte Carlo replications in v1; one seeded demand sample per scenario is enough.
- A Python `3.11+` runtime will be available when implementation starts; the current workspace does not expose a runnable Python interpreter yet.
