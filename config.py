from __future__ import annotations
from typing import Sequence
from dataclasses import dataclass,field
from datetime import datetime
from pathlib import Path



#######     main.py     ########
@dataclass(frozen=True, slots=True)         #Class named in pascal case
class Config:  
    output_dir = Path(__file__).resolve().parent / "rs" #路径
    date = datetime.now().strftime("%y%m%d_%H%M") #日期字符串         
    base_seed: int = 20260601
    seed_count: int = 1 #生成不同的随机需求，用来做重复实验、降低随机性的影响
    pre_alpha: float = 0.5 # prebooking rate
    replication: bool = False #是否复现
    sc: int = 1 # 场景选择：1-需求场景，2-成本场景
    @property
    def scene(self):
        if self.sc == 1:
            return "de"
        elif self.sc == 2:
            return "co"
        return None
    
    o_hotspot: tuple[int, int] = (20, 20) 
    d_hotspot: tuple[int, int] = (70, 70)
    peaks: tuple[int, ...] = (6, 60)
    peak_width_minutes: int = 3 # Gaussian peak width（高斯峰宽）

    span: int = 72 # 时间范围 （time horizon / simulation horizon） /minutes       from7:00 to 19:00
    lambdas: Sequence[int] = tuple(range(10, 100, 20)) # hourly demand intensity（每小时需求强度）
    hs: tuple[float] = (0.5,) # 空间异质性（spatial heterogeneity）
    ht: tuple[float] = (0.5,) # 时间异质性（temporal heterog.eneity）tuple(i/10 for i in range(0, 11))
    service_policy: str = "strict"  #strict/skip
    modes = {
    1: "fixed_route",
    2: "deviated_route",
    3: "drt_rolling_horizon",
    4: "hub_and_spoke",
    }
    max_dev: float = 0.5        # mode2 deviation
    spoke_order = ("north", "east", "south", "west") # pending

@dataclass(frozen=True, slots=True)
class Grid:
    _type: str
    grid: int # size
    grid_len: int # kilos, e.g., 10 for 100x100 grid
    num_routes: int
    #TODO
    # route design parameters
    x_margin: int = 10
    y_margin: int = 20
    half_route_height: int = 10    
    @property
    def routes(self): 
        x_left = self.x_margin
        x_mid = self.grid // 2
        x_right = self.grid - self.x_margin
        y_lower = self.half_route_height
        y_upper = self.grid - 1 - self.half_route_height
        if self.num_routes == 1:
            y_centres = (self.grid // 2,)
        else:
            spacing = (y_upper - y_lower) / (self.num_routes - 1)
            y_centres = tuple(
                round(y_lower + i * spacing)
                for i in range(self.num_routes)
            )

        return tuple(
            (
                (x_left, y),
                (x_left, y + self.half_route_height),
                (x_mid, y + self.half_route_height),
                (x_right, y + self.half_route_height),
                (x_right, y),
                (x_right, y - self.half_route_height),
                (x_mid, y - self.half_route_height),
                (x_left, y - self.half_route_height),
            )
            for y in y_centres
        )
    @property
    def hub(self): 
        return (self.grid // 2, self.grid // 2)

    

@dataclass(frozen=True, slots=True)
class Radial:
    _type: str
    spoke_count: int
    ring_radial: tuple[float, ...] # (5, 10, 15)
    # max_dev: float = 0.5
    #TODO
    # route design parameters
    
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


@dataclass(frozen=True, slots=True)
class Fleet:
    num: int
    cap: int
    freq: int = 20 # 车辆发车频率（分钟/车）
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
    total_departures: int = 0
    net_expenditure: float = 0.0
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
    request_type: str = "real_time"

num_routes = 4
config = Config(lambdas = tuple(range(5, 500, 5)), hs = (0.2,0.5,0.8), ht = (0.2,0.5,0.8), replication = False, sc = 1) # 场景选择：1-需求场景，2-成本场景
nets = Grid(_type = 'grid', grid = 100, grid_len = 1, num_routes = num_routes, x_margin = 10, y_margin = 10)
# nets = Radial(_type = "hub_spoke", spoke_count = 8, ring_radial = (5, 10, 15))
fleet = Fleet(num = num_routes*2, cap = 30)
