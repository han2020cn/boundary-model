

def routes(nets): #grid
    grid_size = nets.grid
    num_routes = nets.num_routes

    spacing = 10
    x_margin = 5
    route_height = 10

    x_left = x_margin
    x_right = grid_size - x_margin

    if num_routes == 1:
        y_above_left = (grid_size // 2 + route_height // 2,)
    else:
        y_above_left = (20,40)

    route_specs = tuple(
        _rectangle_stops(
            x_left=x_left,
            x_right=x_right,
            y_top=y,
            y_bottom=y - route_height,
            spacing=spacing,
        )
        for y in y_above_left
    )

    invalid_nodes = [
        node
        for route in route_specs
        for node in route
        if not (
            0 <= node[0] < grid_size
            and 0 <= node[1] < grid_size
        )
    ]

    if invalid_nodes:
        raise ValueError(
            "Route contains nodes outside the grid: "
            f"{invalid_nodes[:10]}"
        )

    return route_specs


def _rectangle_stops(x_left, x_right, y_top, y_bottom, spacing):
    xs = tuple(range(x_left, x_right + 1, spacing))

    if xs[-1] != x_right:
        raise ValueError("x_right must align with stop spacing")
    if (y_top - y_bottom) % spacing != 0:
        raise ValueError("route height must align with stop spacing")

    top = tuple((x, y_top) for x in xs)
    right = tuple((x_right, y) for y in range(y_top - spacing, y_bottom, -spacing))
    bottom = tuple((x, y_bottom) for x in reversed(xs))
    left = tuple((x_left, y) for y in range(y_bottom + spacing, y_top, spacing))

    return top + right + bottom + left