from __future__ import annotations

import pandas as pd
from dataclasses import dataclass,field
from typing import Sequence
from pathlib import Path
import scenarios as sc
import plt_draw as plt
import demand_generation as dg
import functions as fs
import netx as net


@dataclass(frozen=True, slots=True)         #Class named in pascal case
class Config:           
    base_seed: int = 20260402
    seed_count: int = 1
    pre_alpha: float = 0.5 # prebooking rate
    replication: bool = False #是否复现
    scene: int = 1 # 场景选择：1-需求场景，2-成本场景
    o_hotspot: tuple[int, int] = (2, 2)
    d_hotspot: tuple[int, int] = (7, 7)
    peaks: tuple[int, ...] = (120, 600)
    peak_width_minutes: int = 30 # Gaussian peak width（高斯峰宽）

    span: int = 720 # 时间范围 / 仿真时域（time horizon / simulation horizon）
    lambdas: Sequence[int] = tuple(range(10, 500, 20)) # hourly demand intensity（每小时需求强度）
    hs: tuple[float] = (0.5,) # 空间异质性（spatial heterogeneity）
    ht: tuple[float] = (0.5,) # 时间异质性（temporal heterog.eneity）tuple(i/10 for i in range(0, 11))
    service_policy: str = "strict"  #strict/skip
    modes = {
    1: "fixed_route",
    2: "deviated_route",
    3: "drt_rolling_horizon",
    4: "hub_and_spoke",
    }
    max_dev: float = 0.5
    spoke_order = ("north", "east", "south", "west") # pending

@dataclass(frozen=True, slots=True)
class Nets:

    network_type: str = "grid" # "grid" or "hub_spoke"
    grid: int = 10 # size
    grid_len: int = 1 # 1 miles
    grid_hub: tuple[int, int] = (4, 4)
    # max_dev: float = 0.5
    grid_routes: tuple[tuple[tuple[int, int], ...], ...] = ((
        (1, 5),
        (1, 7),
        (5, 7),
        (9, 7),
        (9, 5),
        (9, 3),
        (5, 3),
        (1, 3),
    ),)
    hub_spoke_hub: str = "hub"
    hub_spoke_routes: tuple[tuple[object, ...], ...] = (
        (
            (15, 0),
            (10, 0),
            (5, 0),
            "hub",
            (5, 4),
            (10, 4),
            (15, 4),
        ),
        (
            (15, 1),
            (10, 1),
            (5, 1),
            "hub",
            (5, 5),
            (10, 5),
            (15, 5),
        ),
        (
            (15, 2),
            (10, 2),
            (5, 2),
            "hub",
            (5, 6),
            (10, 6),
            (15, 6),
        ),
        (
            (15, 3),
            (10, 3),
            (5, 3),
            "hub",
            (5, 7),
            (10, 7),
            (15, 7),
        ),
    )
    spoke_count: int = 8
    ring_radial: tuple[float, ...] = (5, 10, 15)

    @property
    def hub(self):
        if self.network_type == "grid":
            return self.grid_hub
        return self.hub_spoke_hub

@dataclass(frozen=True, slots=True)
class Fleet:
    num: int
    cap: int
    multi_sizes: tuple[int, ...] = (3, 6, 9, 12, 15)
    multi_cap: tuple[int, ...] = (15, 30, 45)
# scenarios_num= len(LAMBDA_LEVELS) * len(HS_LEVELS) * len(HT_LEVELS)


def main(config: Config, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    if config.scene == 1:

        requests, results_frame = sc.demand_scenarios(config, nets, fleet,                                                                  
                                     output_dir,
                                     )
        sc_type = "1"


    if config.scene == 2:
        config.LAMBDA_LEVELS, hs_, ht_ = 40, 0.5, 0.5
        config.seed_count = 5
        results_frame = sc.cost_scenarios(config, nets, fleet, output_dir,
                                     )
        sc_type = "2"
    
    optimals_frame = sc.optimals(results_frame)

    if config.replication: # 如果是复现模式，直接将结果追加到json文件中，否则导出为新的文件
        results_path = sc.extend_json_records(results_frame, output_dir / "com_results.json")
        optimals_path = sc.extend_json_records(optimals_frame, output_dir / "com_optimals.json")
        fs.json_to_excel(results_path)
    else: # 导出新的文件
        results_path = sc.export_files(results_frame, output_dir, sc_type, "rs")
        optimals_path = sc.export_files(optimals_frame, output_dir, sc_type, "ops")
        dg.save_requests(requests, output_dir, "requests.json")
    # print(f"JSON path: {optimals_path}")

    return results_frame, optimals_frame, results_path, optimals_path


if __name__ == "__main__":
    config = Config()
    nets = Nets()
    fleet = Fleet(num = 7, cap = 30)
    output_dir = Path(__file__).resolve().parent / "rs"
    results_frame, optimals_frame, rs_path, optimals_path = main(config, output_dir)
    # results_frame = pd.read_json(output_dir / "demand_rs_260508_1147.json")

    x = "served_requests"
    x1 = "served_requests"
    y = "net_expenditure"
    y1 = "avg_service_time"
    z = "lambda"
    types = ["mode_id"] 
    plot_columns = [
        "mode_id",
        "served_requests",
        "net_expenditure",
        "total_service_time",
    ]

    a = results_frame[["net_expenditure", "total_service_time"]].copy() # dividend

    fr_plot1 = dg.avg_served(results_frame, a, "acceptance") #计算请求的平均值

    plt.plts_2d(fr_plot1,output_dir,y,y1,types)
    plt.plts_2d_pair(   #画图 #plts_3d xyz图, plts_2d xy图, plts_4s 2x2图
        fr_plot1,
        fr_plot1,
        output_dir,
        x,
        x1,
        y,
        y1,
        types,
        left_title=y,
        right_title=y1,
        prebooking_alpha=config.pre_alpha,
    ) 
    

    fr_plot2 = dg.avg_served(results_frame, a, "acceptance") 
    plt.plts_cost_tradeoff(fr_plot2, output_dir, config = config )
