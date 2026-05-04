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
HS_LEVELS = tuple(0.5) # 空间异质性（spatial heterogeneity）
HT_LEVELS = tuple(0.5) # 时间异质性（temporal heterogeneity）i/10 for i in range(0, 11)
SERVICE_POLICY = "strict"
# scenarios_num= len(LAMBDA_LEVELS) * len(HS_LEVELS) * len(HT_LEVELS)



def main(scene: int) -> pd.DataFrame:
    if scene == 1:
        results_frame = sc.demand_scenarios(GRID_SIZE, HORIZON, 
                                     LAMBDA_LEVELS, HS_LEVELS, HT_LEVELS,
                                     BASE_SEED,
                                     service_policy=SERVICE_POLICY,
                                     )
        sc_type = "demand"

    if scene == 2:
        lambda_, hs_, ht_ = 40, 0.5, 0.5
        fleet_sizes = (3, 6, 9, 12, 15)
        capacities = (15, 30, 45)
        seed_count = 5
        results_frame = sc.cost_scenarios(lambda_, hs_, ht_, 
                                     GRID_SIZE, HORIZON, 
                                     fleet_sizes, capacities, 
                                     seed_count, BASE_SEED,
                                     service_policy=SERVICE_POLICY,
                                     )
        sc_type = "cost"
    output_dir = Path("/home/han/from-codex/boundary-model/rs")
    res_type = "rs"
    results_path = sc.export_files(results_frame, output_dir,sc_type,res_type)
    print(f"JSON path: {results_path}")
    optimals_frame = sc.optimals(results_frame)
    res_type = "ops"
    optimals_path = sc.export_files(optimals_frame, output_dir, sc_type,res_type)
    print(f"JSON path: {optimals_path}")
    x = "lambda"
    y = "net_expenditure"
    z = "lambda"
    types = ["mode_id"]
    plt.plts_2d(results_path, x, y, types) #画图
    #plts_3d xyz图, plts_2d xy图, plts_4s 2x2图
    # return json_path

# json to excel: convert files
def json_to_excel(file_path: Path) -> Path:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    output_path = file_path.with_suffix(".xlsx")
    df.to_excel(output_path, index=False)
    

if __name__ == "__main__":
    main(1)
        
