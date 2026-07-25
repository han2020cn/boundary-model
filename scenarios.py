from datetime import datetime
from dataclasses import replace
from pathlib import Path
import random
# from networkx import config
import pandas as pd
from itertools import product
from helpers.config import requests
import helpers.demand_generation as dg
import helpers.fleet_sizing as fleet_sizing
import helpers.common_rule as rule
import helpers.fixedstep as fix
import mode_set

import helpers.netx as net
import helpers.functions as fs
import helpers.hpc as hpc




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
    "operating_time", #所有物理dispatch的总运行时间（veh-min）
    "accepted_deviations",
    # "capacity",
    "total_trips",  #DRT 实际出车次数
    "max_concurrent_trips", #DRT 最大同时运行车辆数
    "vehicle_reuse_ratio", #出车次数 ÷ 车队规模
]

ARTIFACT_POLICIES = {"none", "requests", "all"}


def _build_scenario(config) -> pd.DataFrame:    # 构建场景数据框
    run_seed = (
        random.randint(0, 10**9)
        if config.base_seed is None
        else int(config.base_seed)
    )
    rows = []
    scenario_cases = list(product(config.lambdas, config.hs, config.ht))
    base_scenario_count = len(scenario_cases)
    row_index = 0
    for seed_index in range(config.seed_count):
        rep_num = seed_index + 1
        for index, (lda_value, hs_value, ht_value) in enumerate(scenario_cases):
            # 生成每个场景的随机种子，确保不同场景和不同复制之间的随机性，同时同一场景的不同复制之间有可控的差异
            seed = run_seed + seed_index * base_scenario_count + index 
            rows.append(
                {
                    "_scenario_index": row_index,
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


def _select_scenario_shard(
    scenario_frame: pd.DataFrame,
    shard_id: int,
    shard_count: int,
) -> pd.DataFrame:
    hpc.validate_shard(shard_id, shard_count)
    return scenario_frame.iloc[shard_id::shard_count].copy()


def _artifact_tag(scenario: dict) -> str:
    return (
        f"{scenario['scenario_id']}_rep{int(scenario['rep_num'])}_"
        f"seed{int(scenario['seed'])}"
    )


def _save_request_artifacts(
    requests,
    config,
    nets,
    scenario: dict,
    *,
    artifact_policy: str = "none",
    request_dir: Path,
    artifact_tag: str,
) -> None:
    if artifact_policy not in ARTIFACT_POLICIES:
        raise ValueError(
            f"artifact_policy must be one of {sorted(ARTIFACT_POLICIES)}"
        )
    if artifact_policy == "none":
        return
    if artifact_policy == "all":
        dg.draw_distribution(
            requests,
            nets,
            scenario,
            request_dir,
            artifact_tag,
        )
        dg._draw_timeline(
            requests,
            config.span,
            scenario,
            request_dir,
            artifact_tag,
        )
    dg._requests_save(requests, request_dir, artifact_tag)


def _evaluate_before(
    config, nets, fleet,
    scenario: dict,
    network_context,
    *,
    artifact_policy,
    request_dir: Path | None = None,
    artifact_tag: str | None = None,
):
    required_num = fleet_sizing.required_grid_fleet_num(nets, network_context, fleet)
    sized_fleet = replace(fleet, num = required_num)
    sized_scenario = {
        **scenario,
        "fleet_max": sized_fleet.num,
        "capacity": sized_fleet.cap,
    }
    loop_context = fix.build_context(network_context,config, nets, sized_fleet)
    #requests
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
    try:
        baseline = mode_set.build_baseline(
            config,
            nets,
            sized_fleet,
            network_context,
            sized_scenario,
            requests,
            loop_context,
        )
    except ValueError as exc:
        raise ValueError(
            f"Failed to build baseline for scenario"
        ) from exc
    filter_requests = [assignment["request"] for assignment in baseline["assignments"]]
    if request_dir is None:
        request_dir = Path(__file__).resolve().parent / "rs" / "requests"
    if artifact_tag is None:
        artifact_tag = datetime.now().strftime("%m%d_%H%M%S")
    _save_request_artifacts(
        filter_requests,
        config,
        nets,
        scenario,
        artifact_policy=artifact_policy,
        request_dir=request_dir,
        artifact_tag=artifact_tag,
    )
    return sized_fleet, sized_scenario, baseline, filter_requests


def evaluate_all(
    config, nets, fleet,
    network_context,
    scenario: dict,
    requests,
    baseline,
) -> list[dict]:        
    service_policy = config.service_policy
    mode_1_result = baseline["result"]
    benchmark_expenditure = float(mode_1_result["net_expenditure"])
    result_rows = [mode_1_result]

    result_rows.append(
        mode_set.evaluate_2(
            config, nets, fleet,
            baseline,
            requests,
            scenario,
            network_context,
            benchmark_expenditure,
            service_policy=service_policy,
        )
    )
    result_rows.append(
        mode_set.evaluate_3(
            config, nets, fleet,
            requests,
            scenario,
            network_context,
            benchmark_expenditure,
            service_policy=service_policy,
        )
    )
    return result_rows



def demand_scenario(
    config,
    nets,
    fleet,
    *,
    shard_id: int = 0,
    shard_count: int = 1,
    run_id: str | None = None,
    output_root: Path | None = None,
    artifact_policy,
    resume: bool = False,
) -> pd.DataFrame:
    hpc.validate_shard(shard_id, shard_count)
    if artifact_policy not in ARTIFACT_POLICIES:
        raise ValueError(
            f"artifact_policy must be one of {sorted(ARTIFACT_POLICIES)}"
        )

    full_scenario_frame = _build_scenario(config)
    scenario_frame = _select_scenario_shard(
        full_scenario_frame,
        shard_id,
        shard_count,
    )
    result_rows: list[dict] = []
    manifest = None
    manifest_path = None

    if run_id is None:
        if shard_id != 0 or shard_count != 1 or output_root is not None or resume:
            raise ValueError(
                "run_id is required for sharding, custom output_root, or resume"
            )
        init_path = config.output_dir / f"{config.scene}_{config.date}_init.json"
        request_dir = Path(__file__).resolve().parent / "rs" / "requests"
        completed_indices: set[int] = set()
    else:
        run_id = hpc.validate_run_id(run_id)
        if output_root is None:
            output_root = config.output_dir 
        run_dir = Path(output_root) / run_id
        shard_dir = run_dir / f"shard_{shard_id:02d}"
        init_path = shard_dir / f"init_{shard_id:02d}.json"
        manifest_path = shard_dir / "manifest.json"
        request_dir = shard_dir / "requests"
        expected_manifest = hpc.new_manifest(
            run_id=run_id,
            shard_id=shard_id,
            shard_count=shard_count,
            config_fingerprint=hpc.configuration_fingerprint(config, nets, fleet),
            assigned_scenarios=scenario_frame.to_dict(orient="records"),
            result_columns=RESULT_COLUMNS,
        )
        manifest_exists = manifest_path.exists()
        result_exists = init_path.exists()
        if (manifest_exists or result_exists) and not resume:
            raise FileExistsError(
                f"shard output already exists; use --resume: {shard_dir}"
            )
        if result_exists and not manifest_exists:
            raise ValueError(f"shard result exists without manifest: {shard_dir}")

        if manifest_exists:
            manifest = hpc.load_manifest(manifest_path)
            hpc.validate_manifest(manifest, expected_manifest, path=manifest_path)
            result_rows = hpc.load_result_records(
                init_path,
                RESULT_COLUMNS,
                allow_missing=True,
            )
            completed_indices = hpc.completed_scenario_indices(
                result_rows,
                manifest["assigned_scenarios"],
            )
            manifest_completed = {
                int(index)
                for index in manifest["completed_scenario_indices"]
            }
            if not manifest_completed.issubset(completed_indices):
                raise ValueError(
                    "manifest records completed scenarios missing from init.json"
                )
            if manifest_completed != completed_indices:
                manifest["completed_scenario_indices"] = sorted(completed_indices)
                fs.write_json(manifest, manifest_path)
        else:
            manifest = expected_manifest
            completed_indices = set()
            fs.write_json(manifest, manifest_path)
            fs.write_json([], init_path)

    if len(completed_indices) == len(scenario_frame):
        return pd.DataFrame(result_rows, columns=RESULT_COLUMNS)

    # One network and one shortest-path cache are reused by all scenarios in a shard.
    network_context = net.build_network_context(nets)
    for scenario in scenario_frame.to_dict(orient="records"):
        scenario_index = int(scenario["_scenario_index"])
        if scenario_index in completed_indices:
            print(
                f"Skipping completed scenario index={scenario_index} "
                f"in shard {shard_id}"
            )
            continue
        #loops_context
        sized_fleet, sized_scenario, baseline,filter_requests = _evaluate_before(config, nets, fleet,
                                                                scenario,
                                                                network_context,
                                                                artifact_policy=artifact_policy,
                                                                request_dir=request_dir,
                                                                artifact_tag=(
                                                                    _artifact_tag(scenario)
                                                                    if run_id is not None
                                                                    else None
                                                                ),
                                                                )
        #一次添加4个mode的结果
        scenario_results = (
            evaluate_all(   
                config, nets, sized_fleet,
                network_context,
                sized_scenario,
                filter_requests,
                baseline,
            )
        )
        result_rows.extend(scenario_results)
        # 每完成一个场景就覆盖保存一次，避免之后 break 时丢失已有结果。
        init_results = pd.DataFrame(result_rows, columns=RESULT_COLUMNS)
        fs.export_json(init_results, init_path)
        completed_indices.add(scenario_index)
        if manifest is not None and manifest_path is not None:
            manifest["completed_scenario_indices"] = sorted(completed_indices)
            fs.write_json(manifest, manifest_path)

    init_results = pd.DataFrame(result_rows, columns=RESULT_COLUMNS)
    if run_id is None:
        # Preserve the existing local-run derived output.
        infes_path = config.output_dir / f"{config.scene}_{config.date}_infeasible.json"
        infeasible_results = _infeasible_result(init_results)
        fs.export_json(infeasible_results, infes_path)
    return init_results


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
        #loops_context
        sized_fleet, sized_scenario, baseline = _evaluate_before()
        #一次添加4个mode的结果
        scenario_results = (
            evaluate_all(   
                config, nets, sized_fleet,
                network_context,
                sized_scenario,
                requests,
                baseline,
            )
        )
        result_rows.extend(scenario_results)

    return  pd.DataFrame(result_rows, columns=RESULT_COLUMNS)


def mean_result(init_results: pd.DataFrame):
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
    "operating_time",
    "accepted_deviations",
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

def _infeasible_result(init_results: pd.DataFrame):
    infeasible_results = init_results.loc[init_results["feasible"] == False].copy()
    return infeasible_results
