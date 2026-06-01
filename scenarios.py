import json
from pathlib import Path
import random
import pandas as pd
from itertools import product
import demand_generation as dg
import mode_set

import netx as net
import functions as fs

from mode_set import (
    _evaluate_1,
    _evaluate_2,
    _evaluate_3,
    _evaluate_4,
)


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
    "objective_value",
    "fleet_size",
    "capacity",
    "total_trips",
    "max_concurrent_trips",
    "vehicle_reuse_ratio",
]


def build_scenarios(config) -> pd.DataFrame:    # 构建场景数据框
    run_seed = (
        random.randint(0, 10**9)
        if config.base_seed is None
        else int(config.base_seed)
    )

    rows = []
    scenario_cases = list(product(config.lambdas, config.hs, config.ht))
    base_scenario_count = len(scenario_cases)
    seed_count = int(getattr(config, "seed_count"))
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


def request_types( # 预订或实时
    requests: list,
    scenario: dict,
    prebooking_alpha: float,
    *,
    fixed_seed: bool,
) -> list:
    request_type_seed = int(scenario["seed"]) if fixed_seed else None
    return dg._request_types_assign(
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
    input_dir = output_dir / "re_demand"
    result_rows = []

    if config.replication and not input_dir.exists():
        raise FileNotFoundError(f"Replication directory not found: {input_dir}")

    for index, scenario in enumerate(scenario_frame.to_dict(orient="records")):
        if config.replication:  #是否复现
            requests = dg.load_requests(config.rep)
        else:
            requests = dg._requests_generate(
                config,
                nets,
                scenario,
                network_context,
            )
        _validate_requests_in_graph(requests, generated_graph)  #validation
        requests = request_types(   # pre-booking or real-time
            requests,
            scenario,
            config.pre_alpha,
            fixed_seed= False,
        )

        loop_results = (
            evaluate_all(   #一次添加4个mode的结果
                requests,
                scenario,
                generated_graph,
                service_policy = config.service_policy,
            )
        )
        result_rows.extend(loop_results)
        loop_frame = pd.DataFrame(loop_results, columns=RESULT_COLUMNS)
        loop_output_path = output_dir / f"{config.scene}_result_{config.date}.json"
        if index == 0:
            fs.export_json(loop_frame, loop_output_path) # 导出结果数据
        else:
            fs.extend_json(loop_frame, loop_output_path) # 追加结果数据

    return requests, pd.DataFrame(result_rows, columns=RESULT_COLUMNS)


def cost_scenarios(
        config,
        nets,
        fleet,
        output_dir: Path,
        ) -> pd.DataFrame:
    network_context = net.build_network_context(nets)
    mode_set.configure_runtime(config, nets, fleet, network_context)
    scenario_frame = build_scenarios(config)
    input_dir = output_dir / "re_demand"
    generated_graph = network_context.graph
    result_rows = []

    if config.replication and not input_dir.exists():
        raise FileNotFoundError(f"Replication directory not found: {input_dir}")

    for scenario in scenario_frame.to_dict(orient="records"):
        if config.replication:
            requests = dg.load_requests(config.rep)
        else:
            requests = dg._requests_generate(
                config,
                nets,
                scenario,
                network_context,
            )
#TODO
        _validate_requests_in_graph(requests, generated_graph)
        requests = request_types(
            requests,
            scenario,
            config.pre_alpha,
            fixed_seed= False,
        )

        result_rows.extend(
            evaluate_all(   #一次添加4个mode的结果
                requests,
                scenario,
                generated_graph,
                service_policy = config.service_policy,
            )
        )


    return requests, pd.DataFrame(result_rows, columns=RESULT_COLUMNS)






'''
选出每个 scenario 的 optimal
'''
def optimals(results: pd.DataFrame | Path):

    if isinstance(results, pd.DataFrame):
        frame = results.copy()
    else:
        frame = pd.read_json(Path(results))

    min_objective = frame.groupby("scenario_id")["objective_value"].transform("min")
    optimal_df = frame[frame["objective_value"] == min_objective].copy()

    return optimal_df

