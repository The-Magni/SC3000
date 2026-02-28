from queue import PriorityQueue
from typing import Dict, List

from data import load_data


def ucs(
    start_node: str,
    end_node: str,
    Cost: Dict[str, int],
    Dist: Dict[str, float],
    G: Dict[str, List[str]],
):
    path_cost = {node: float("inf") for node in G}
    energy_cost = {node: 0 for node in G}
    pi = {node: "" for node in G}
    path_cost[start_node] = 0.0
    visited = set()
    pq = PriorityQueue()
    pq.put((path_cost[start_node], start_node))

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
                pq.put((path_cost[v], v))

    return pi, path_cost[end_node], energy_cost[end_node]


if __name__ == "__main__":
    _, Cost, Dist, G = load_data("./data")
    start_node, end_node = "1", "50"
    current = end_node
    shortest_path = end_node
    pi, shortest_dist, energy_cost = ucs(start_node, end_node, Cost, Dist, G)

    while current != start_node:
        shortest_path = f"{pi[current]}->" + shortest_path
        current = pi[current]

    print(f"Shortest path: {shortest_path}.")
    print(f"Shortest distance: {shortest_dist}.")
    print(f"Total energy cost: {energy_cost}.")
