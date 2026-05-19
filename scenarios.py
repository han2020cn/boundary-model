import json
from pathlib import Path
import random
from types import SimpleNamespace
import pandas as pd
from itertools import product
from datetime import datetime
from demand_generation import generate_requests
import mode_set

import netx as net

from mode_set import (
    _evaluate_1,
    _evaluate_2,
    _evaluate_3,
    _evaluate_4,
)
import demand_generation as dg

RESULT_COLUMNS = [
    "scenario_id",
    "lambda",
    "hs",
    "ht",
    "seed",
    "replication_id",
    "mode_id",
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
    scenario_cases = list(product(config.lambdas, config.hs, config.ht))
    base_scenario_count = len(scenario_cases)
    seed_count = int(getattr(config, "seed_count", 1))
    row_index = 1
    for seed_index in range(seed_count):
        replication_id = seed_index + 1
        for index, (lda_value, hs_value, ht_value) in enumerate(scenario_cases):
            seed = run_seed + seed_index * base_scenario_count + index
            rows.append(
                {
                    "scenario_id": (
                        f"S{row_index:03d}_r{replication_id:02d}"
                        f"_l{lda_value}_hs{hs_value:.1f}_ht{ht_value:.1f}"
                        f"_seed{seed}"
                    ),
                    "lambda": int(lda_value),
                    "hs": float(hs_value),
                    "ht": float(ht_value),
                    "seed": seed,
                    "replication_id": replication_id,
                }
            )
            row_index += 1
    return pd.DataFrame(rows)


def build_fleets(fleet) -> pd.DataFrame:
    fleet_sizes = getattr(fleet, "sizes", None)
    if fleet_sizes is None:
        fleet_sizes = (fleet.num,)
    capacities = getattr(fleet, "capacities", None)
    if capacities is None:
        capacities = (fleet.cap,)
    rows = [
        {
            "fleet_size": int(size),
            "capacity": int(capacity),
        }
        for size, capacity in product(fleet_sizes, capacities)
    ]
    return pd.DataFrame(rows)


def _config_horizon(config) -> int:
    horizon = getattr(config, "horizon", None)
    if horizon is None:
        horizon = config.span
    return int(horizon)


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


def _validate_requests_in_graph(requests: list, graph) -> None:
    missing = []
    for request in requests:
        if request.origin not in graph:
            missing.append((request.request_id, "origin", request.origin))
        if request.destination not in graph:
            missing.append((request.request_id, "destination", request.destination))
    if missing:
        details = ", ".join(
            f"request {request_id} {field}={node!r}"
            for request_id, field, node in missing[:10]
        )
        if len(missing) > 10:
            details += f", ... {len(missing) - 10} more"
        raise ValueError(f"requests contain nodes that are not in graph: {details}")


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
    mode_1_result = _evaluate_1(requests, scenario, graph)
    result_rows = [mode_1_result]

    benchmark_expenditure = (
        float(mode_1_result["net_expenditure"])
        if mode_1_result["feasible"]
        else None
    )
    if benchmark_expenditure is None:
            print('infeasible')

    result_rows.append(
        _evaluate_2(
            requests,
            scenario,
            graph,
            benchmark_expenditure,
            service_policy=service_policy,
        )
    )
    result_rows.append(
        _evaluate_3(
            requests,
            scenario,
            graph,
            benchmark_expenditure,
            service_policy=service_policy,
        )
    )
    result_rows.append(
        _evaluate_4(
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
    network_context = net.build_network_context(nets)
    mode_set.configure_runtime(config, nets, fleet, network_context)
    generated_graph = network_context.graph
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
                config,
                nets,
                scenario,
                network_context,
            )
        _validate_requests_in_graph(requests, generated_graph)
        requests = _assign_scenario_request_types(
            requests,
            scenario,
            config.pre_alpha,
            fixed_seed=not config.replication,
        )
        if not config.replication:
            # dg.save_requests(requests, output_dir, _request_file_name(scenario))
            pass

        result_rows.extend(
            evaluate_all(   #一次添加4个mode的结果
                requests,
                scenario,
                generated_graph,
                service_policy = config.service_policy,
            )
        )

    return requests, pd.DataFrame(result_rows, columns=RESULT_COLUMNS)


def cost_scenarios(
        config,
        nets,
        fleet,
        output_dir: Path,
        ) -> pd.DataFrame:
    network_context = net.build_network_context(nets)
    mode_set.configure_runtime(config, nets, fleet, network_context)
    generated_graph = network_context.graph
    scenario_frame = build_scenarios(config)
    fleet_frame = build_fleets(fleet)
    result_rows = []
    input_dir = output_dir / "re_demand"
    scenario_index = 1

    if config.replication and not input_dir.exists():
        raise FileNotFoundError(f"Replication directory not found: {input_dir}")

    for scenario in scenario_frame.to_dict(orient="records"):
        if config.replication:
            requests = _load_replicated_requests(scenario, input_dir)
        else:
            requests = generate_requests(
                config,
                nets,
                scenario,
                network_context,
            )
        _validate_requests_in_graph(requests, generated_graph)
        requests = _assign_scenario_request_types( #为每个请求分配类型（预订或实时）
            requests,
            scenario,
            config.pre_alpha,
            fixed_seed=not config.replication,
        )

        for fleet_case in fleet_frame.to_dict(orient="records"):
            fleet_size = int(fleet_case["fleet_size"])
            capacity = int(fleet_case["capacity"])
            cost_scenario = {
                **scenario,
                **fleet_case,
                "scenario_id": (
                    f"C{scenario_index:03d}_l{scenario['lambda']:g}"
                    f"_hs{scenario['hs']:.1f}_ht{scenario['ht']:.1f}"
                    f"_f{fleet_size}_c{capacity}_seed{scenario['seed']}"
                ),
                "fleet_size": fleet_size,
                "capacity": capacity,
            }

            runtime_fleet = SimpleNamespace(num=fleet_size, cap=capacity)
            mode_set.configure_runtime(config, nets, runtime_fleet, network_context)
            result_rows.extend(
                evaluate_all(
                    requests,
                    cost_scenario,
                    generated_graph,
                    service_policy=config.service_policy,
                )
            )

            scenario_index += 1

    mode_set.configure_runtime(config, nets, fleet, network_context)
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
选出每个 scenario 的 optimal
'''
def optimals(results_frame: pd.DataFrame):
    # 读取结果
    df = results_frame
    
    # 只保留可行方案
    # feasible_df = df[df["feasible"] == True].copy()

        # 先按 scenario_id 分组，再按 unserved_requests、avg_service_time、net_expenditure 排序
    df = df.copy()
    denominator = df["served_requests"].where(df["served_requests"] > 0)
    df["_service_time_per_served"] = (
        df["total_service_time"] / denominator
    ).fillna(0.0)

    # 先按 scenario_id 分组，再按 unserved_requests、total_service_time / served_requests、net_expenditure 排序
    df = df.sort_values(
        by=["scenario_id", "unserved_requests", "_service_time_per_served", "net_expenditure"]
    )

    # 每个 scenario 取第一条 = optimal mode
    optimal_df = df.groupby("scenario_id", as_index=False).first()
    optimal_df = optimal_df.drop(columns=["_service_time_per_served"])
    return optimal_df
