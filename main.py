from __future__ import annotations
import json
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
SERVICE_POLICY = "strict"
# scenarios_num= len(LAMBDA_LEVELS) * len(HS_LEVELS) * len(HT_LEVELS)



def main(scene: int) -> pd.DataFrame:
    if scene == 1:
        results_frame = sc.demand_scenarios(GRID_SIZE, HORIZON, 
                                     LAMBDA_LEVELS, HS_LEVELS, HT_LEVELS,
                                     service_policy=SERVICE_POLICY,
                                     )
    if scene == 2:
        LAMBDA_LEVELS, HS_LEVELS, HT_LEVELS = 40, 0.5, 0.5
        fleet_sizes = (3, 6, 9, 12, 15)
        capacities = (15, 30, 45)
        seed_count = 5
        results_frame = sc.cost_scenarios(LAMBDA_LEVELS, HS_LEVELS, HT_LEVELS, 
                                     GRID_SIZE, HORIZON, 
                                     fleet_sizes, capacities, 
                                     seed_count, BASE_SEED,
                                     service_policy=SERVICE_POLICY,
                                     )
    output_dir = Path("/home/han/from-codex/boundary-model/rs")
    results_path = sc.results_export(results_frame, output_dir)
    print(f"JSON path: {results_path}")
    optimals_path = sc.optimals(results_path)
    print(f"JSON path: {optimals_path}")
    x = "net_expenditure"
    y = "total_service_time"
    z = "lambda"
    types = ["mode_id"]
    plt.plts_2d(results_path, x, y, types) #画图
    #plts_3d xyz图, plts_2d xy图, plts_4s 2x2图
    # return json_path

def json_to_excel(file_path: Path) -> Path:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    output_path = file_path.with_suffix(".xlsx")
    df.to_excel(output_path, index=False)
    
if __name__ == "__main__":
    main(2)
        
