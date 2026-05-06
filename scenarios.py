import json
from pathlib import Path
import random
import pandas as pd
from itertools import product
from datetime import datetime
from demand_generation import generate_requests
import mode_set

import netx as net

from mode_set import (
    evaluate_1,
    evaluate_2,
    evaluate_3,
    evaluate_4,
)
import demand_generation as dg

RESULT_COLUMNS = [
    "scenario_id",
    "lambda",
    "hs",
    "ht",
    "seed",
    "mode_id",
    "mode_name",
    "feasible",
    "feasibility_reason",
    "total_requests",
    "served_requests",
    "unserved_requests",
    "benchmark_expenditure",
    "net_expenditure", # cost
    "total_wait",
    "total_walk",
    "total_onboard",
    "total_service_time",
    "avg_wait",
    "avg_walk",
    "avg_onboard",
    "avg_service_time",
    "fleet_size",
    "capacity",
]


def build_scenarios(config) -> pd.DataFrame:
    run_seed = (
        random.randint(0, 10**9)
        if config.base_seed is None
        else int(config.base_seed)
    )

    rows = []
    for index, (lda_value, hs_value, ht_value) in enumerate(
        product(config.lambdas, config.hs, config.ht),
        start=1,
    ):
        rows.append(
            {
                "scenario_id": f"S{index:02d}_l{lda_value}_hs{hs_value:.1f}_ht{ht_value:.1f}",
                "lambda": int(lda_value),
                "hs": float(hs_value),
                "ht": float(ht_value),
                "seed": run_seed + index - 1,
            }
        )
    return pd.DataFrame(rows)


def build_fleets(fleet) -> tuple[tuple[int, int], ...]:
    fleet_sizes = getattr(fleet, "sizes", (fleet.num,))
    capacities = getattr(fleet, "capacities", (fleet.cap,))
    return tuple((int(size), int(capacity)) for size, capacity in product(fleet_sizes, capacities))


def _request_file_name(scenario: dict) -> str:
    return (
        f"lambda{scenario['lambda']:g}"
        f"_hs{scenario['hs']:g}"
        f"_ht{scenario['ht']:g}"
        f"_seed{scenario['seed']}.json"
    )


def _load_replicated_requests(scenario: dict, input_dir: Path) -> list:
    request_path = input_dir / _request_file_name(scenario)
    if not request_path.exists():
        raise FileNotFoundError(f"Replication request file not found: {request_path}")
    
    return dg.load_requests(request_path)

def extend_json_records(frame: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
    else:
        records = []

    records.extend(json.loads(frame.to_json(orient="records")))
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)

    return output_path
'''
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
'''


def evaluate_all(
    requests: list,
    scenario: dict,
    graph,
    service_policy: str = "strict",
) -> list[dict]:
    mode_1_result = evaluate_1(requests, scenario, graph)
    result_rows = [mode_1_result]

    benchmark_expenditure = (
        float(mode_1_result["net_expenditure"])
        if mode_1_result["feasible"]
        else None
    )
    if benchmark_expenditure is None:
            print('infeasible')

    result_rows.append(
        evaluate_2(
            requests,
            scenario,
            graph,
            benchmark_expenditure,
            service_policy=service_policy,
        )
    )
    result_rows.append(
        evaluate_3(
            requests,
            scenario,
            graph,
            benchmark_expenditure,
            service_policy=service_policy,
        )
    )
    result_rows.append(
        evaluate_4(
            requests,
            scenario,
            graph,
            benchmark_expenditure,
            service_policy=service_policy,
        )
    )
    return result_rows


def _assign_scenario_request_types(
    requests: list,
    scenario: dict,
    prebooking_alpha: float,
    *,
    fixed_seed: bool = True,
) -> list:
    request_type_seed = int(scenario["seed"]) if fixed_seed else None
    return dg.assign_request_types(
        requests,
        alpha=prebooking_alpha,
        seed=request_type_seed,
    )


def demand_scenarios(
    config,
    nets,
    fleet,
    output_dir: Path,
) -> pd.DataFrame:
    mode_set.configure_runtime(config, nets, fleet)
    grid_graph = net.build_grid_graph(nets.grid)
    scenario_frame = build_scenarios(config)
    result_rows = []
    input_dir = output_dir / "re_demand"
    requests = []

    if config.replication and not input_dir.exists():
        raise FileNotFoundError(f"Replication directory not found: {input_dir}")

    for scenario in scenario_frame.to_dict(orient="records"):
        if config.replication:
            requests = _load_replicated_requests(scenario, input_dir)
        else:
            requests = generate_requests(
                lambda_value = float(scenario["lambda"]),
                hs = float(scenario["hs"]),
                ht = float(scenario["ht"]),
                seed = int(scenario["seed"]),
                grid_size = nets.grid,
                horizon = config.horizon,
            )
        requests = _assign_scenario_request_types(
            requests,
            scenario,
            config.pre_alpha,
            fixed_seed=not config.replication,
        )
        if not config.replication:
            dg.save_requests(requests, output_dir, _request_file_name(scenario))

        result_rows.extend(
            evaluate_all(   #一次添加4个mode的结果
                requests,
                scenario,
                grid_graph,
                service_policy = config.service_policy,
            )
        )

    return requests, pd.DataFrame(result_rows, columns=RESULT_COLUMNS)


def cost_scenarios(
        config,
        nets,
        fleet,
        ) -> tuple[pd.DataFrame, Path]:
    mode_set.configure_runtime(config, nets, fleet)
    graph = net.build_grid_graph(nets.grid)
    result_rows = []
    scenario_index = 1

    lambda_value = int(config.lambdas[0])
    hs_value = float(config.hs[0])
    ht_value = float(config.ht[0])

    for seed_offset in range(config.seed_count):
        seed = config.base_seed + seed_offset
        requests = generate_requests(
            lambda_value=float(lambda_value),
            hs=hs_value,
            ht=ht_value,
            seed=int(seed),
            grid_size=nets.grid,
            horizon=config.horizon,
        )
        requests = _assign_scenario_request_types( #为每个请求分配类型（预订或实时）
            requests,
            {"seed": int(seed)},
            config.pre_alpha,
        )

        for fleet_size, capacity in build_fleets(fleet):
            scenario = {
                "scenario_id": (
                    f"C{scenario_index:03d}_l{lambda_value:g}"
                    f"_hs{hs_value:.1f}_ht{ht_value:.1f}"
                    f"_f{fleet_size}_c{capacity}_seed{seed}"
                ),
                "lambda": lambda_value,
                "hs": hs_value,
                "ht": ht_value,
                "seed": int(seed),
                "fleet_size": int(fleet_size),
                "capacity": int(capacity),
            }

            original_fleet_size = fleet.num
            original_vehicle_capacity = fleet.cap
            try:
                fleet.num = int(fleet_size)
                fleet.cap = int(capacity)
                result_rows.extend(
                    evaluate_all(
                        requests,
                        scenario,
                        graph,
                        service_policy=config.service_policy,
                    )
                )
            finally:
                fleet.num = original_fleet_size
                fleet.cap = original_vehicle_capacity

            scenario_index += 1

    results_frame = pd.DataFrame(result_rows, columns=RESULT_COLUMNS)
    return results_frame


def export_files(_frame: pd.DataFrame, output_dir: Path,scenario_type: str,results_type: str) ->  Path:
    #output_dir.mkdir(parents=True, exist_ok=True)
    #csv_path = output_dir / "scenario_results.csv"
    #results_frame.to_csv(csv_path, index=False)
    date = datetime.now().strftime("%y%m%d_%H%M")
    json_path = output_dir / f"{scenario_type}_{results_type}_{date}.json"
    json_records = json.loads(_frame.to_json(orient="records"))
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(json_records, handle, ensure_ascii=False, indent=2)

    return json_path

'''
选出每个 scenario 的 optimal
'''
def optimals(results_frame: pd.DataFrame):
    # 读取结果
    df = results_frame
    
    # 只保留可行方案
    # feasible_df = df[df["feasible"] == True].copy()

    # 先按 scenario_id 分组，再按 unserved_requests、avg_service_time、net_expenditure 排序
    df = df.sort_values(
        by=["scenario_id", "unserved_requests", "avg_service_time", "net_expenditure"]
    )

    # 每个 scenario 取第一条 = optimal mode
    optimal_df = df.groupby("scenario_id", as_index=False).first()
    return optimal_df
