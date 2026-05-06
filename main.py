from __future__ import annotations
import json
import pandas as pd
from dataclasses import dataclass,field
from typing import Sequence
from pathlib import Path
import scenarios as sc
import plt_draw as plt

@dataclass(frozen=True, slots=True)
class config:
    base_seed: int = 20260402
    seed_count: int = 3
    pre_alpha: float = 0.5 # prebooking rate
    replication: bool = True #是否复现
    scene: int = 1 # 场景选择：1-需求场景，2-成本场景

    span: int = 180 # 时间范围 / 仿真时域（time horizon / simulation horizon）
    lambdas: Sequence[int] = tuple(range(1, 101, 5)) # 需求强度或到达率（demand intensity / arrival rate）
    hs: tuple[float] = (0.5,) # 空间异质性（spatial heterogeneity）
    ht: tuple[float] = (0.5,) # 时间异质性（temporal heterog.eneity）tuple(i/10 for i in range(0, 11))
    service_policy: str = "strict"  #strict/skip
    modes = {
    1: "fixed_route",
    2: "deviated_route",
    3: "drt_rolling_horizon",
    4: "hub_and_spoke",
    }
    spoke_order = ("north", "east", "south", "west") # pending

@dataclass(frozen=True, slots=True)
class nets:

    grid: int = 10
    hub: tuple = (4, 4)
    fixed_stops: tuple[tuple[int, int], ...] = (
    (1, 5),
    (1, 7),
    (5, 7),
    (9, 7),
    (9, 5),
    (9, 3),
    (5, 3),
    (1, 3),
    )
    spoke_count: int = 8,
    ring_radii: tuple[float, ...] = (5, 10, 15),

@dataclass(frozen=True, slots=True)
class fleet:
    num = 7
    cap = 30
# scenarios_num= len(LAMBDA_LEVELS) * len(HS_LEVELS) * len(HT_LEVELS)

def main(scene: int, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    if scene == 1:
        requests, results_frame = sc.demand_scenarios(config,nets,fleet,                                                                  
                                     output_dir,
                                     )
        sc_type = "demand"

    if scene == 2:
        config.LAMBDA_LEVELS, hs_, ht_ = 40, 0.5, 0.5
        fleet_sizes = (3, 6, 9, 12, 15)
        capacities = (15, 30, 45)
        configseed_count = 5
        results_frame = sc.cost_scenarios(config, nets,fleet, 
                                     )
        sc_type = "cost"
    
    optimals_frame = sc.optimals(results_frame)

    if config.replication:
        results_path = sc.extend_json_records(results_frame, output_dir / "com_results.json")
        optimals_path = sc.extend_json_records(optimals_frame, output_dir / "com_optimals.json")
    else:
        results_path = sc.export_files(results_frame, output_dir, sc_type, "rs")
        optimals_path = sc.export_files(optimals_frame, output_dir, sc_type, "ops")
    # print(f"JSON path: {optimals_path}")

    return results_frame, optimals_frame, results_path, optimals_path

# json to excel: convert files
def json_to_excel(file_path: Path) -> Path:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    output_path = file_path.with_suffix(".xlsx")
    df.to_excel(output_path, index=False)
    

if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent / "rs"
    results_frame, optimals_frame, rs_path, optimals_path = main(config.scene, output_dir)

    x = "served_requests"
    x1 = "served_requests"
    y = "net_expenditure"
    y1 = "avg_service_time"
    z = "lambda"
    types = ["mode_id"] 
    plt.plts_2d(results_frame,output_dir,y,y1,types)
    plt.plts_2d_pair(
        results_frame,
        results_frame,
        output_dir,
        x,
        x1,
        y,
        y1,
        types,
        left_title=y,
        right_title=y1,
        prebooking_alpha=config.pre_alpha,
    ) #画图
    #plts_3d xyz图, plts_2d xy图, plts_4s 2x2图
