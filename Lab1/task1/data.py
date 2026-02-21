import json
import os
from typing import Dict, List


def load_json_data(path: str):
    with open(path, "r") as file:
        data = json.load(file)
    return data


def load_data(data_dir: str):
    Coord: Dict[str, List[int]] = load_json_data(os.path.join(data_dir, "Coord.json"))
    Cost: Dict[str, int] = load_json_data(os.path.join(data_dir, "Cost.json"))
    Dist: Dict[str, float] = load_json_data(os.path.join(data_dir, "Dist.json"))
    G: Dict[str, List[str]] = load_json_data(os.path.join(data_dir, "G.json"))
    return Coord, Cost, Dist, G


if __name__ == "__main__":
    pass
