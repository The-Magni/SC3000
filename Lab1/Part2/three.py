import random
from collections import defaultdict

import pandas as pd
from utils import ACTIONS, GOAL, START


def q_learning(num_episodes: int = 500):
    Q = defaultdict(float)
    policy = []
    for ep in range(num_episodes):
        s = START
        while s != GOAL:
            policy_s = [0.0 for _ in range(len(ACTIONS))]
            a_star = Q[s].index(max(Q[s]))
