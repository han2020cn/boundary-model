from __future__ import annotations
import pandas as pd
from pathlib import Path
import scenarios as sc
import plt_draw as plt

BASE_SEED = 20260402
GRID_SIZE = 10
HORIZON = 180 # 时间范围 / 仿真时域（time horizon / simulation horizon）
LAMBDA_LEVELS = tuple(range(20, 61, 10)) # 需求强度或到达率（demand intensity / arrival rate）
HS_LEVELS = tuple(i/10 for i in range(0, 11)) # 空间异质性（spatial heterogeneity）
HT_LEVELS = tuple(i/10 for i in range(0, 11)) # 时间异质性（temporal heterogeneity）
# scenarios_num= len(LAMBDA_LEVELS) * len(HS_LEVELS) * len(HT_LEVELS)



def main(scene: int) -> pd.DataFrame:
    if scene == 1:
        results_frame = sc.demand_scenarios(GRID_SIZE, HORIZON, 
                                     LAMBDA_LEVELS, HS_LEVELS, HT_LEVELS, 
                                     )
    else:
        LAMBDA_LEVELS, HS_LEVELS, HT_LEVELS = 40, 0.5, 0.5
        fleet_sizes=(3, 6, 9, 12, 15)
        capacities=(15, 30, 45) 
        seed_count=5
        results_frame = sc.cost_scenarios(LAMBDA_LEVELS, HS_LEVELS, HT_LEVELS, 
                                     GRID_SIZE, HORIZON, 
                                     fleet_sizes, capacities, 
                                     seed_count, BASE_SEED,
                                     )
    output_dir  = Path("/home/han/from-codex/boundary-model/rs")
    json_path = sc.results_export(results_frame, output_dir)
    return json_path



if __name__ == "__main__":
        json_path = main(2)
        print(f"JSON path: {json_path}")
        optimals_path = sc.optimals(json_path)
        plt.optimals_3d(optimals_path)
