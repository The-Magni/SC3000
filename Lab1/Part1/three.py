import math
import heapq
from typing import Dict, List, Tuple, Optional

from data import load_data

EARTH_RADIUS_M = 6371000.0


def make_edge_id_uv(u: str, v: str) -> str:
    return u + "," + v


def coord_to_latlon_deg(coord_xy: List[int], scale: float = 1e6) -> Tuple[float, float]:
    # Coord example: {"1": [-73530767, 41085396]} => lon=-73.530767, lat=41.085396
    lon = coord_xy[0] / scale
    lat = coord_xy[1] / scale
    return lat, lon


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def make_haversine_distance_to_goal_heuristic(goal: str, Coord: Dict[str, List[int]], scale: float = 1e6):
    goal_lat, goal_lon = coord_to_latlon_deg(Coord[goal], scale)

    def h(v: str) -> float:
        lat, lon = coord_to_latlon_deg(Coord[v], scale)
        return haversine_m(lat, lon, goal_lat, goal_lon)

    return h


def dominates(a_dist: float, a_energy: int, b_dist: float, b_energy: int) -> bool:
    # a dominates b if no worse in both and strictly better in at least one
    return (a_dist <= b_dist and a_energy <= b_energy) and (a_dist < b_dist or a_energy < b_energy)


def reconstruct_path_from_label_ids(pred: Dict[int, Optional[int]], node_of: Dict[int, str], goal_id: int) -> List[str]:
    path = []
    cur: Optional[int] = goal_id
    while cur is not None:
        path.append(node_of[cur])
        cur = pred[cur]
    path.reverse()
    return path


def astar_rcspp_multilabel(
    start: str,
    goal: str,
    energy_budget: int,
    Coord: Dict[str, List[int]],
    Cost: Dict[str, int],
    Dist: Dict[str, float],
    G: Dict[str, List[str]],
) -> Tuple[List[str], float, int]:
    """
    Task 3:
    Minimize total distance subject to total energy <= energy_budget.
    Multi-label A* (Pareto labels per node) + haversine heuristic.
    Returns: (best_path_nodes, best_distance, best_energy)
    """
    h = make_haversine_distance_to_goal_heuristic(goal, Coord)

    # labels[node] = list of nondominated (dist, energy)
    labels: Dict[str, List[Tuple[float, int]]] = {node: [] for node in G}

    # label storage for backtracking
    pred: Dict[int, Optional[int]] = {}
    node_of: Dict[int, str] = {}
    dist_of: Dict[int, float] = {}
    energy_of: Dict[int, int] = {}

    next_id = 0

    def new_label(node: str, dist: float, energy: int, parent: Optional[int]) -> int:
        nonlocal next_id
        lid = next_id
        next_id += 1
        pred[lid] = parent
        node_of[lid] = node
        dist_of[lid] = dist
        energy_of[lid] = energy
        return lid

    # heap entries: (f=dist+h, dist, energy, node, label_id)
    heap: List[Tuple[float, float, int, str, int]] = []

    start_id = new_label(start, 0.0, 0, None)
    labels[start].append((0.0, 0))
    heapq.heappush(heap, (h(start), 0.0, 0, start, start_id))

    best_goal_id: Optional[int] = None
    best_goal_dist = float("inf")
    best_goal_energy = 0

    while heap:
        f_u, d_u, e_u, u, lid_u = heapq.heappop(heap)

        # bound: if cannot beat current best goal distance, skip
        if best_goal_id is not None and d_u + h(u) >= best_goal_dist:
            continue

        if u == goal:
            if d_u < best_goal_dist:
                best_goal_id = lid_u
                best_goal_dist = d_u
                best_goal_energy = e_u
            continue

        for v in G.get(u, []):
            k = make_edge_id_uv(u, v)
            edge_d = Dist.get(k)
            edge_e = Cost.get(k)
            if edge_d is None or edge_e is None:
                continue

            nd = d_u + edge_d
            ne = e_u + edge_e
            if ne > energy_budget:
                continue

            # if any existing label dominates (nd,ne), discard
            dominated = False
            for (d_old, e_old) in labels[v]:
                if dominates(d_old, e_old, nd, ne):
                    dominated = True
                    break
            if dominated:
                continue

            # remove labels dominated by the new label
            new_list = []
            for (d_old, e_old) in labels[v]:
                if not dominates(nd, ne, d_old, e_old):
                    new_list.append((d_old, e_old))
            new_list.append((nd, ne))
            labels[v] = new_list

            lid_v = new_label(v, nd, ne, lid_u)
            heapq.heappush(heap, (nd + h(v), nd, ne, v, lid_v))

    if best_goal_id is None:
        return [], float("inf"), 0

    path = reconstruct_path_from_label_ids(pred, node_of, best_goal_id)
    return path, best_goal_dist, best_goal_energy


if __name__ == "__main__":
    Coord, Cost, Dist, G = load_data("./data")

    start_node = "1"
    end_node = "50"
    total_energy_budget = 287932

    path, shortest_dist, energy_cost = astar_rcspp_multilabel(
        start=start_node,
        goal=end_node,
        energy_budget=total_energy_budget,
        Coord=Coord,
        Cost=Cost,
        Dist=Dist,
        G=G,
    )
    if not path:
        print("Shortest path: .")
        print("Shortest distance: inf.")
        print("Total energy cost: 0.")
    else:
        shortest_path_str = "->".join(path)
        print(f"Shortest path: {shortest_path_str}.")
        print(f"Shortest distance: {shortest_dist}.")
        print(f"Total energy cost: {energy_cost}.")