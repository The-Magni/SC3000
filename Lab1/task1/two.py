"""
Reference about Resource Constraint Shortest Path Problem:
https://www.emergentmind.com/topics/resource-constrained-shortest-path-problem-rcspp
"""

from queue import PriorityQueue
from typing import Dict, List, Tuple

from data import load_data


def constraint_satisfying_ucs(
    start_node: str,
    end_node: str,
    total_energy: int,
    Cost: Dict[str, int],
    Dist: Dict[str, float],
    G: Dict[str, List[str]],
):
    path_cost: Dict[tuple[str, int], float] = {}  # (node, energy): distance cost
    pi: Dict[Tuple[str, int], Tuple[str, int]] = {}
    path_cost[(start_node, 0)] = 0.0
    pq: PriorityQueue[Tuple[float, Tuple[str, int]]] = (
        PriorityQueue()
    )  # priority = distance, key = (node, energy)
    domninating_labels: Dict[str, List[Tuple[float, int]]] = {
        node: [] for node in G
    }  # store the possible shortest path and lowest energy of a node
    pq.put((0.0, (start_node, 0)))

    while not pq.empty():
        _, (u, energy) = pq.get()
        if (u, energy) not in path_cost:
            continue  # this state already get pruned
        if u == end_node:
            return pi, path_cost[(u, energy)], energy
        for v in G[u]:
            energy_v = energy + Cost[f"{u},{v}"]
            path_cost_v = path_cost[(u, energy)] + Dist[f"{u},{v}"]
            if energy_v <= total_energy and (
                path_cost.get((v, energy_v)) is None
                or path_cost_v < path_cost[(v, energy_v)]
            ):
                is_pruned = False
                new_labels_v = []
                for possible_dist, possible_energy in domninating_labels[v]:
                    if path_cost_v >= possible_dist and energy_v >= possible_energy:
                        is_pruned = True
                    if possible_dist < path_cost_v or possible_energy < energy_v:
                        new_labels_v.append(
                            (possible_dist, possible_energy)
                        )  # pruning the old labels

                if is_pruned:
                    path_cost.pop((v, energy_v), None)
                    pi.pop((v, energy_v), None)
                else:
                    path_cost[(v, energy_v)] = path_cost_v
                    pi[(v, energy_v)] = (u, energy)
                    pq.put((path_cost_v, (v, energy_v)))
                    domninating_labels[v] = new_labels_v
                    domninating_labels[v].append(
                        (path_cost_v, energy_v)
                    )  # add new label

    # no path found to goal node
    return pi, float("inf"), 0


if __name__ == "__main__":
    _, Cost, Dist, G = load_data("./data")
    start_node = "1"
    end_node = "50"
    total_energy = 287932
    current: str = end_node
    shortest_path = end_node
    pi, shortest_dist, energy_cost = constraint_satisfying_ucs(
        start_node, end_node, total_energy, Cost, Dist, G
    )
    current_energy = energy_cost

    while current != start_node:
        shortest_path = f"{pi[(current, current_energy)][0]}->" + shortest_path
        current, current_energy = pi[(current, current_energy)]

    print(f"Shortest path: {shortest_path}.")
    print(f"Shortest distance: {shortest_dist}.")
    print(f"Total energy cost: {energy_cost}.")
