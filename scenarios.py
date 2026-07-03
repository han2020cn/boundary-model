import json
from dataclasses import replace
from pathlib import Path
import random
import pandas as pd
from itertools import product
import helpers.demand_generation as dg
import helpers.fleet_sizing as fleet_sizing
import mode_set

import helpers.netx as net
import helpers.functions as fs

from mode_set import (
    evaluate_1,
    evaluate_2,
    evaluate_3,
    evaluate_4
)


RESULT_COLUMNS = [
    "scenario_id",
    "seed",
    "lambda",
    "ht",
    "hs",
    "rep_num",
    "mode_id",
    "objective_value",
    "net_expenditure", # cost
    "benchmark_expenditure",
    "feasible",
    "feasibility_reason",
    "total_requests",
    "served_requests",
    "unserved_requests",

    "total_wait",
    "total_walk",
    "total_onboard",
    "total_service_time",
    "fleet_max",
    # "capacity",
    "total_trips",  #DRT 实际出车次数
    "max_concurrent_trips", #DRT 最大同时运行车辆数
    "vehicle_reuse_ratio", #出车次数 ÷ 车队规模
]


def _build_scenario(config) -> pd.DataFrame:    # 构建场景数据框
    run_seed = (
        random.randint(0, 10**9)
        if config.base_seed is None
        else int(config.base_seed)
    )
    rows = []
    scenario_cases = list(product(config.lambdas, config.hs, config.ht))
    base_scenario_count = len(scenario_cases)
    row_index = 1
    for seed_index in range(config.seed_count):
        rep_num = seed_index + 1
        for index, (lda_value, hs_value, ht_value) in enumerate(scenario_cases):
            # 生成每个场景的随机种子，确保不同场景和不同复制之间的随机性，同时同一场景的不同复制之间有可控的差异
            seed = run_seed + seed_index * base_scenario_count + index 
            rows.append(
                {
                    "scenario_id": (
                        f"L{lda_value}_ht{ht_value:.1f}_hs{hs_value:.1f}"
                    ),
                    "lambda": int(lda_value),
                    "ht": float(ht_value),
                    "hs": float(hs_value),
                    "seed": seed,
                    "rep_num": rep_num,
                }
            )
            row_index += 1
    return pd.DataFrame(rows)


def _evaluate_all(
    config, nets, fleet,
    requests: list,
    scenario: dict,
    network_context,
    service_policy: str = "strict",
) -> list[dict]:
    required_num = fleet_sizing.required_grid_fleet_num(nets, network_context, fleet)
    sized_fleet = replace(fleet, num=required_num)
    sized_scenario = {
        **scenario,
        "fleet_max": sized_fleet.num,
        "capacity": sized_fleet.cap,
    }

    mode_1_result = evaluate_1(
        config,
        nets,
        sized_fleet,
        requests,
        sized_scenario,
        network_context,
    )
    result_rows = [mode_1_result]

    if mode_1_result["feasible"]:
        benchmark_expenditure = float(mode_1_result["net_expenditure"])
    else: raise ValueError("benchmark expenditure unavable")


    result_rows.append(
        evaluate_2(
            config, nets, sized_fleet,
            requests,
            sized_scenario,
            network_context,
            benchmark_expenditure,
            service_policy=service_policy,
        )
    )
    result_rows.append(
        evaluate_3(
            config, nets, sized_fleet,
            requests,
            sized_scenario,
            network_context,
            benchmark_expenditure,
            service_policy=service_policy,
        )
    )
    # result_rows.append(
    #     evaluate_4(
    #         config, nets, sized_fleet,
    #         requests,
    #         sized_scenario,
    #         network_context,
    #         benchmark_expenditure,
    #         service_policy=service_policy,
    #     )
    # )
    return result_rows



def demand_scenario(
    config,
    nets,
    fleet,
) -> pd.DataFrame:
    network_context = net.build_network_context(nets) 
    scenario_frame = _build_scenario(config)
    result_rows = []
    for index, scenario in enumerate(scenario_frame.to_dict(orient="records")):
        if config.replication:  #是否复现
            requests = dg.load_requests(config.rep)
        else:
            requests = dg.requests_generate(
                config,
                nets,
                scenario,
                network_context,
                fixed_seed= False, # False:不用scenario的seed
            )

        scenario_results = (
            _evaluate_all(   #一次添加4个mode的结果
                config, nets, fleet,
                requests,
                scenario,
                network_context,
                service_policy = config.service_policy,
            )
        )
        result_rows.extend(scenario_results)
    init_results = pd.DataFrame(result_rows, columns=RESULT_COLUMNS)
    init_path = config.output_dir / f"{config.scene}_init_{config.date}.json"
    fs.export_json(init_results, init_path) # 导出init结果数据
    mean_results = _mean_result(init_results)
    mean_path = config.output_dir / f"{config.scene}_final_{config.date}.json"
    fs.export_json(mean_results, mean_path) # 导出mean结果数据
    optimals_results = _optimal(init_results)
    optimals_path = config.output_dir / f"{config.scene}_optimal_{config.date}.json"    
    fs.export_json(optimals_results, optimals_path) # 导出optimal结果数据
    return requests, init_results


def cost_scenario(
        config,
        nets,
        fleet,
        ) -> pd.DataFrame:
    network_context = net.build_network_context(nets)
    scenario_frame = _build_scenario(config)
    result_rows = []
    for scenario in scenario_frame.to_dict(orient="records"):
        if config.replication:
            requests = dg.load_requests(config.rep)
        else:
            requests = dg.requests_generate(
                config,
                nets,
                scenario,
                network_context,
                fixed_seed= False, # 场景内不同rep之间的请求类型分配保持一致
            )
        result_rows.extend(
            _evaluate_all(   #一次添加4个mode的结果
                config, nets, fleet,
                requests,
                scenario,
                network_context,
                service_policy = config.service_policy,
            )
        )
    return requests, pd.DataFrame(result_rows, columns=RESULT_COLUMNS)



'''
选出每个 scenario 的 optimal
'''
def _optimal(results: pd.DataFrame | Path):

    if isinstance(results, pd.DataFrame):
        frame = results.copy()
    else:
        frame = pd.read_json(Path(results))

    min_objective = frame.groupby("scenario_id")["objective_value"].transform("min")
    optimal_df = frame[frame["objective_value"] == min_objective].copy()

    return optimal_df

def _mean_result(init_results: pd.DataFrame):
    feasible_results = init_results.loc[init_results["feasible"] == True].copy()  
    mean_columns = [
    "objective_value",
    "benchmark_expenditure",
    "net_expenditure",
    "total_requests",
    "served_requests",
    "unserved_requests",
    "total_wait",
    "total_walk",
    "total_onboard",
    "total_service_time",
    "fleet_max",
    "total_trips",
    "max_concurrent_trips",
    "vehicle_reuse_ratio",
    ]  
    aggregation = {
    "lambda": "first",
    "ht": "first",
    "hs": "first",
    "rep_num": "nunique",
    }
    aggregation.update({
    column: "mean"
    for column in mean_columns
    })
    mean_results = (
    feasible_results
    .groupby(["scenario_id", "mode_id"], as_index=False)
    .agg(aggregation)
    .rename(columns={"rep_num": "rep_count"})
    )
    return mean_results