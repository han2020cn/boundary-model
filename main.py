from __future__ import annotations

import pandas as pd

import scenarios as sc
import helpers.functions as fs
#导入class
from helpers.config import Config, Grid, Radial, Fleet

# 场景选择：1-需求场景，2-成本场景
config = Config(lambdas = tuple(range(10, 90, 20)), hs = (0.2,0.8), ht = (0.2,0.8), replication = False, sc = 1) 
nets = Grid(_type = 'grid', grid = 10, grid_len = 100, num_routes = 2)
# nets = Radial(_type = "hub_spoke", spoke_count = 8, ring_radial = (5, 10, 15))
fleet = Fleet(cap = 30)

def main(config) -> tuple[pd.DataFrame, pd.DataFrame]:
    if config.scene ==  "de":
        requests, results_frame = sc.demand_scenario(config, nets, fleet,          
                                     )
    if config.scene ==  "co":
        requests, results_frame = sc.cost_scenario(config, nets, fleet,
                                     )
    return results_frame

def local_result() -> pd.DataFrame:
    file_name = "de_result_260602_1927.json"
    results_path = config.output_dir / file_name
    sc, result_type, config.date, time = file_name.removesuffix(".json").split("_")
    optimals_results = sc.optimals(results_path)
    optimals_path = config.output_dir / f"{sc}_optimal_{config.date}_{time}.json"
    fs.export_json(optimals_results, optimals_path)


if __name__ == "__main__":
    
    main(config)
    # fs.transfer_json_to_excel(config.output_dir/"de_result_260608_1559.json")
    
    # print("Processing complete.")
    # input("Press Enter to continue...")
    # print("Continuing program...")
