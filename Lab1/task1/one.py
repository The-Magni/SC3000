import math
from queue import PriorityQueue
from typing import Dict, List

from data import load_data


def build_heuristic(end_node: str, Coord: Dict[str, List[int]]) -> Dict[str, float]:
    h = {}
    for node in Coord:
        h[node] = math.sqrt(
            (Coord[node][0] - Coord[end_node][0]) ** 2
            + (Coord[node][1] - Coord[end_node][1]) ** 2
        )
    return h


def a_star(
    start_node: str,
    end_node: str,
    Coord: Dict[str, List[int]],
    Cost: Dict[str, int],
    Dist: Dict[str, float],
    G: Dict[str, List[str]],
):
    h = build_heuristic(end_node, Coord)
    path_cost = {node: float("inf") for node in G}
    energy_cost = {node: 0 for node in G}
    pi = {node: "" for node in G}
    path_cost[start_node] = 0.0
    visited = set()
    pq = PriorityQueue()
    pq.put((path_cost[start_node] + h[start_node], start_node))

    while not pq.empty():
        _, u = pq.get()
        if u in visited:
            continue  # already visited this node
        visited.add(u)
        if u == end_node:
            break
        for v in G[u]:
            if path_cost[u] + Dist[f"{u},{v}"] < path_cost[v]:
                path_cost[v] = path_cost[u] + Dist[f"{u},{v}"]
                pi[v] = u
                energy_cost[v] = energy_cost[u] + Cost[f"{u},{v}"]
                pq.put((path_cost[v] + h[v], v))

    return pi, path_cost[end_node], energy_cost[end_node]


if __name__ == "__main__":
    Coord, Cost, Dist, G = load_data("./data")
    start_node, end_node = "1", "50"
    current = end_node
    shortest_path = end_node
    pi, shortest_dist, energy_cost = a_star(start_node, end_node, Coord, Cost, Dist, G)

    while current != start_node:
        shortest_path = f"{pi[current]}->" + shortest_path
        current = pi[current]

    print(f"Shortest path: {shortest_path}.")
    print(f"Shortest distance: {shortest_dist}.")
    print(f"Total energy cost: {energy_cost}.")
