from __future__ import annotations
from typing import Sequence
from dataclasses import dataclass,field
from datetime import datetime
from pathlib import Path

from helpers.types import GridNode, NetworkNode


Hotspot = tuple[float, float]
HotspotConfig = Hotspot | tuple[Hotspot, ...]


#######     main.py     ########
@dataclass(frozen=True, slots=True)         #Class named in pascal case
class Config:  
    ##### scenario parameters
    lambdas: Sequence[int]  # hourly demand intensity（每小时需求强度）
    hs: tuple[float]  # spatial 
    ht: tuple[float]  # temporal
    replication: bool  #是否复现
    sc: int # 场景选择：1-需求场景，2-成本场景
    seed_count: int # 生成不同的随机需求，用来做重复实验、降低随机性的影响
    pre_alpha: float = 0.5 # prebooking rate
    span: int = 300 # 时间范围 （time horizon / simulation horizon） /minutes       
    peaks: tuple[int, ...] = (60,)
    peak_width_minutes: int = 30 # Gaussian peak width（高斯峰宽）
    o_hotspot: HotspotConfig = ((10, 10), (10, 30))
    d_hotspot: HotspotConfig = ((40, 10), (40, 30))
    ##### 导出设定
    output_dir = Path(__file__).resolve().parents[1]/ "rs" #路径
    date = f"{datetime.now().strftime('%y%m%d_%H%M')}" #日期字符串 
    ##### 重复设定        
    base_seed: int = 20260710
    ##### pedestrian parameter
    walk_speed: float = 33  # m/minute 2km/h
    ##### 复现的demand路径
    rep = Path("rs/requests/requests_0707_140602.json")
    @property
    def scene(self):
        if self.sc == 1:
            return "de"
        elif self.sc == 2:
            return "co"
        return None

    service_policy: str = "strict"  #strict/skip
    modes = {
    1: "fixed",
    2: "deviated",
    3: "drt",
    4: "hub_spoke",
    }

@dataclass(frozen=True, slots=True)
class Grid:
    _type: str
    grid: int # size
    grid_len: int # m
    num_routes: int
    @property
    def hub(self):
        return (0, self.grid // 2)
    max_dev: float = 500        # mode2 deviation 多少m

    

@dataclass(frozen=True, slots=True)
class Radial:
    # _type: str
    spoke_count: int
    ring_radial: tuple[float, ...] # (5, 10, 15)
    #TODO
    
    routes: tuple[tuple[object, ...], ...] = (
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

    hub_spoke_hub: str = "hub"
    @property
    def hub(self):
        if self._type == "grid":
            return self.grid_hub
        return self.hub_spoke_hub


@dataclass(frozen=False, slots=True)
class Fleet:
    cap: int
    num: int = 0
    freq: int = 30 # 车辆发车频率（分钟/车）
    speed: float = 420 # 车辆速度（m/minute）25km/h
    multi_sizes: tuple[int, ...] = (3, 6, 9, 12, 15)
    multi_cap: tuple[int, ...] = (15, 30, 45)
# scenarios_num= len(LAMBDA_LEVELS) * len(HS_LEVELS) * len(HT_LEVELS)


################  mode_set.py    ################

# loop route for mode 1 and 2
@dataclass(frozen=True, slots=True)
class LoopContext:
    id: str
    nodes: tuple[GridNode, ...]
    length: int
    fixed_stop_indices: dict[GridNode, int]
    optional_stops: tuple[GridNode, ...]
    optional_anchor_indices: dict[GridNode, int]
    vehicle_offsets: dict[int, int]
    headway: float | None = None # 车辆发车间隔（分钟/车），如果有的话

    @property
    def route_id(self) -> str:
        return self.id

    @property
    def route_nodes(self) -> tuple[GridNode, ...]:
        return self.nodes

    @property
    def route_length(self) -> int:
        return self.length

# spoke paths for mode 4
@dataclass(frozen=True, slots=True)
class SpokeVehicle:
    vehicle_id: int
    spoke_name: str
    first_departure: int

# results
@dataclass
class requests:
    request_id: int
    origin: GridNode
    destination: GridNode
    departure_time: int

@dataclass(slots=True)
class ModeAccumulator:
    served_requests: int = 0
    total_wait: float = 0.0
    total_walk: float = 0.0
    total_onboard: float = 0.0
    total_travel_distance: float = 0.0
    operating_time: float = 0.0
    total_trips: int = 0
    net_expenditure: float = 0.0
    accepted_deviations: int = 0
    total_trips: int | None = None
    max_concurrent_trips: int | None = None
    vehicle_reuse_ratio: float | None = None

#######    demand_generation.py    ########
@dataclass(frozen=True, slots=True)
class TripRequest:
    request_id: int
    origin: NetworkNode
    destination: NetworkNode
    departure_time: int
    request_type: int # "pre_booking" = 1
