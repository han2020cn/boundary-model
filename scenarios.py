import json
from pathlib import Path
import random
import pandas as pd
from itertools import product
from datetime import datetime
from demand_generation import generate_requests
import mode_set
from mode_set import (
    RESULT_COLUMNS, # column names for the results DataFrame
    build_grid_graph,
    evaluate_mode_1,
    evaluate_mode_2,
    evaluate_mode_3,
    evaluate_mode_4,
)

def build_scenario_frame(lda: list, hs: list, ht: list,
                         run_seed: int | None = None, 
                         ) -> pd.DataFrame:
    if run_seed is None:
        run_seed = random.randint(0, 10**9)

    rows = []
    for index, (lda, hs, ht) in enumerate(
        product(lda, hs, ht),
        start=1,
    ):
        rows.append(
            {
                "scenario_id": f"S{index:02d}_l{lda}_hs{hs:.1f}_ht{ht:.1f}",
                "lambda": int(lda),
                "hs": float(hs),
                "ht": float(ht),
                "seed": run_seed + index - 1,
            }
        )
    return pd.DataFrame(rows)

''''
def _benchmark_mode_infeasible_result(
    mode_id: int,
    requests: list,
    scenario: dict,
) -> dict:
    return mode_set._finalize_result(
        mode_id=mode_id,
        scenario=scenario,
        total_requests=len(requests),
        served_requests=0,
        benchmark_expenditure=None,
        net_expenditure=0.0,
        total_wait=0.0,
        total_walk=0.0,
        total_onboard=0.0,
        feasible=False,
        feasibility_reason="benchmark_mode_infeasible",
    )
''''

def evaluate_all(
    requests: list,
    scenario: dict,
    graph,
) -> list[dict]:
    mode_1_result = evaluate_mode_1(requests, scenario, graph)
    result_rows = [mode_1_result]

    benchmark_expenditure = (
        float(mode_1_result["net_expenditure"])
        if mode_1_result["feasible"]
        else None
    )
    if benchmark_expenditure is None:
        result_rows.extend(
            print('infeasible')
        return result_rows

    result_rows.append(
        evaluate_mode_2(requests, scenario, graph, benchmark_expenditure)
    )
    result_rows.append(
        evaluate_mode_3(requests, scenario, graph, benchmark_expenditure)
    )
    result_rows.append(
        evaluate_mode_4(requests, scenario, graph, benchmark_expenditure)
    )
    return result_rows


def demand_scenarios(size, span, lda, hs, ht) -> pd.DataFrame:
    graph = build_grid_graph(size)
    scenario_frame = build_scenario_frame(lda=lda, hs=hs, ht=ht)
    result_rows = []

    for scenario in scenario_frame.to_dict(orient="records"):
        requests = generate_requests(
            lambda_value=float(scenario["lambda"]),
            hs=float(scenario["hs"]),
            ht=float(scenario["ht"]),
            seed=int(scenario["seed"]),
            grid_size=size,
            horizon=span,
        )

        result_rows.extend(evaluate_all(requests, scenario, graph))

    return pd.DataFrame(result_rows, columns=RESULT_COLUMNS)


def cost_scenarios(
        lambda_value: float,
        hs: float,
        ht: float,
        grid_size: int,
        horizon: int,
        fleet_sizes: tuple[int, ...],
        capacities: tuple[int, ...],
        seed_count: int,
        base_seed: int,
        output_dir: Path = Path.cwd(),
        ) -> tuple[pd.DataFrame, Path]:
    graph = build_grid_graph(grid_size)
    result_rows = []
    scenario_index = 1

    for seed_offset in range(seed_count):
        seed = base_seed + seed_offset
        requests = generate_requests(
            lambda_value=float(lambda_value),
            hs=float(hs),
            ht=float(ht),
            seed=int(seed),
            grid_size=grid_size,
            horizon=horizon,
        )

        for fleet_size, capacity in product(fleet_sizes, capacities):
            scenario = {
                "scenario_id": (
                    f"C{scenario_index:03d}_l{lambda_value:g}"
                    f"_hs{hs:.1f}_ht{ht:.1f}"
                    f"_f{fleet_size}_c{capacity}_seed{seed}"
                ),
                "lambda": int(lambda_value),
                "hs": float(hs),
                "ht": float(ht),
                "seed": int(seed),
                "fleet_size": int(fleet_size),
                "capacity": int(capacity),
            }

            original_fleet_size = mode_set.FLEET_SIZE
            original_vehicle_capacity = mode_set.VEHICLE_CAPACITY
            try:
                mode_set.FLEET_SIZE = int(fleet_size)
                mode_set.VEHICLE_CAPACITY = int(capacity)
                result_rows.extend(evaluate_all(requests, scenario, graph))
            finally:
                mode_set.FLEET_SIZE = original_fleet_size
                mode_set.VEHICLE_CAPACITY = original_vehicle_capacity

            scenario_index += 1

    results_frame = pd.DataFrame(result_rows, columns=RESULT_COLUMNS)
    return results_frame


def results_export(results_frame: pd.DataFrame, output_dir: Path) ->  Path:
    #output_dir.mkdir(parents=True, exist_ok=True)
    #csv_path = output_dir / "scenario_results.csv"
    #results_frame.to_csv(csv_path, index=False)
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"results_{date}.json"
    json_records = json.loads(results_frame.to_json(orient="records"))
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(json_records, handle, ensure_ascii=False, indent=2)

    return json_path

'''
选出每个 scenario 的 optimal
'''
def optimals(json_path: Path):
    # 读取结果
    df = pd.read_json(json_path)
    
    # 只保留可行方案
    feasible_df = df[df["feasible"] == True].copy()

    # 先按 scenario_id 分组，再按 avg_service_time、net_expenditure 排序
    feasible_df = feasible_df.sort_values(
        by=["scenario_id", "avg_service_time", "net_expenditure"]
    )

    # 每个 scenario 取第一条 = optimal mode
    optimal_df = feasible_df.groupby("scenario_id", as_index=False).first()
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonpath_opt = json_path.with_name(f"optimals_{date}.json")
    optimal_df.to_json(jsonpath_opt, orient="records", force_ascii=False, indent=2)
    return str(jsonpath_opt)
