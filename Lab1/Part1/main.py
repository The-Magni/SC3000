from data import load_data
from one import ucs
from three import astar_rcspp_multilabel
from two import constraint_satisfying_ucs

if __name__ == "__main__":
    Coord, Cost, Dist, G = load_data("./data")
    start_node = "1"
    end_node = "50"
    total_energy_budget = 287932
    while True:
        try:
            option = int(
                input("""Please choose 1,2, or 3\n
                    1. Run part 1.\n
                    2. Run part 2.\n
                    3. Run part 3.\n
                    Any other key: Exit application.\n""")
            )
        except ValueError:
            print("Exited")
            break
        match option:
            case 1:
                pi, shortest_dist, energy_cost = ucs(
                    start_node, end_node, Cost, Dist, G
                )
                current = end_node
                shortest_path = end_node
                while current != start_node:
                    shortest_path = f"{pi[current]}->" + shortest_path
                    current = pi[current]

                print(f"Shortest path: {shortest_path}.")
                print(f"Shortest distance: {shortest_dist}.")
                print(f"Total energy cost: {energy_cost}.")
            case 2:
                pi, shortest_dist, energy_cost = constraint_satisfying_ucs(
                    start_node, end_node, total_energy_budget, Cost, Dist, G
                )
                current = end_node
                shortest_path = end_node
                current_energy = energy_cost

                while current != start_node:
                    shortest_path = (
                        f"{pi[(current, current_energy)][0]}->" + shortest_path
                    )
                    current, current_energy = pi[(current, current_energy)]

                print(f"Shortest path: {shortest_path}.")
                print(f"Shortest distance: {shortest_dist}.")
                print(f"Total energy cost: {energy_cost}.")

            case 3:
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
            case _:
                print("Exited")
                break
