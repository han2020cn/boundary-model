

def routes(nets): #grid
    grid_size = nets.grid
    num_routes = nets.num_routes
    x_margin: int = 1
    y_margin: int = 2
    half_height: int = 1 

    x_left = x_margin
    x_mid = grid_size // 2
    x_right = grid_size - x_margin
    y_lower = half_height
    y_upper = grid_size - 1 - half_height

    if num_routes == 1:
        y_centres = (grid_size // 2,)
    else:
        spacing = (y_upper - y_lower) / (num_routes - 1)
        y_centres = tuple(
            round(y_lower + i * spacing)
            for i in range(num_routes)
        )

    route_specs = tuple(
        (
            (x_left, y),
            (x_left, y + half_height),
            (x_mid, y + half_height),
            (x_right, y + half_height),
            (x_right, y),
            (x_right, y - half_height),
            (x_mid, y - half_height),
            (x_left, y - half_height),
        )
        for y in y_centres
    )

    invalid_nodes = [
        node
        for route in route_specs
        for node in route
        if not (0 <= node[0] < grid_size and 0 <= node[1] < grid_size)
    ]
    if invalid_nodes:
        raise ValueError(
            "route contains nodes outside the grid: "
            f"{invalid_nodes[:10]}"
        )

    return route_specs
