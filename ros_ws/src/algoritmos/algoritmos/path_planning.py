import numpy as np
from scipy.ndimage import binary_dilation

def inflate_map(map_data, radius_cells):
    y, x = np.ogrid[-radius_cells:radius_cells+1, -radius_cells:radius_cells+1]
    kernel = (x**2 + y**2) <= radius_cells**2
    inflated = binary_dilation(map_data, structure=kernel)
    return inflated

def get_neighborhood(cell, shape):
    neighbors = []
    y, x = cell
    rows, cols = shape
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < rows and 0 <= nx < cols:
                neighbors.append([ny, nx])
    return neighbors

def get_heuristic(cell, goal):
    dy = cell[0] - goal[0]
    dx = cell[1] - goal[1]
    return np.sqrt(dy**2 + dx**2)

def get_edge_cost(parent, child, inflated):
    cy, cx = child
    py, px = parent

    if inflated[cy, cx]:
        return np.inf

    return np.sqrt((cy - py)**2 + (cx - px)**2)

def a_star(inflated, start, goal):
    """
    inflated: np.array 2D booleano (True = bloqueado)
    start: (col, row) en grilla
    goal: (col, row) en grilla
    Devuelve lista de (col, row) desde start hasta goal, o None.
    """
    start = np.array([start[1], start[0]])  # (col,row) -> (row,col)
    goal = np.array([goal[1], goal[0]])

    if inflated[start[0], start[1]] or inflated[goal[0], goal[1]]:
        return None

    costs = np.ones(inflated.shape) * np.inf
    closed_flags = np.zeros(inflated.shape)
    predecessors = -np.ones(inflated.shape + (2,), dtype=np.int32)

    heuristic = np.zeros(inflated.shape)
    for x in range(inflated.shape[0]):
        for y in range(inflated.shape[1]):
            heuristic[x, y] = get_heuristic([x, y], goal)

    costs[start[0], start[1]] = 0
    parent = start

    while not np.array_equal(parent, goal):
        open_costs = np.where(closed_flags == 1, np.inf, costs) + heuristic

        x, y = np.unravel_index(open_costs.argmin(), open_costs.shape)

        if open_costs[x, y] == np.inf:
            break

        parent = np.array([x, y])
        closed_flags[x, y] = 1

        for neighbor in get_neighborhood(parent, inflated.shape):
            ny, nx = neighbor
            edge = get_edge_cost(parent, neighbor, inflated)

            if closed_flags[ny, nx] == 1 or edge == np.inf:
                continue

            new_cost = costs[parent[0], parent[1]] + edge

            if new_cost < costs[ny, nx]:
                costs[ny, nx] = new_cost
                predecessors[ny, nx] = parent

    if not np.array_equal(parent, goal):
        return None

    # Reconstruir camino
    path = []
    current = goal
    while predecessors[current[0], current[1]][0] >= 0:
        path.append((current[1], current[0]))  # (row,col) -> (col,row)
        current = predecessors[current[0], current[1]]
    path.append((start[1], start[0]))
    path.reverse()

    return path