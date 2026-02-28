import random
from typing import Dict

from utils import (
    ACTIONS,
    DELTA,
    GOAL,
    GOAL_REWARD,
    STEP_COST,
    in_bounds,
    is_blocked,
    is_terminal,
)

EPS = 0.1  # fixed epsilon for ε-greedy, as required

# Perpendicular directions for each intended action
PERP = {
    "U": ("L", "R"),
    "D": ("L", "R"),
    "L": ("U", "D"),
    "R": ("U", "D"),
}


def valid_next(s, a):
    """Deterministic move for action a with boundary/roadblock handling."""
    if is_terminal(s):
        return s
    dx, dy = DELTA[a]
    cand = (s[0] + dx, s[1] + dy)
    if in_bounds(cand) and not is_blocked(cand):
        return cand
    return s


def step_stochastic_sample(s, intended_a):
    """
    Sample one transition from the stochastic environment:
    - 0.8: intended action
    - 0.1: perpendicular left
    - 0.1: perpendicular right
    Reward:
    - -1 per step
    - +10 on entering GOAL (terminal)
    """
    if is_terminal(s):
        return s, 0, True

    r = random.random()
    if r < 0.8:
        executed_a = intended_a
    elif r < 0.9:
        executed_a = PERP[intended_a][0]
    else:
        executed_a = PERP[intended_a][1]

    s_next = valid_next(s, executed_a)
    reward = GOAL_REWARD if s_next == GOAL else STEP_COST
    done = s_next == GOAL
    return s_next, reward, done


def epsilon_greedy_probs(Q, s, eps=EPS):
    nA = len(ACTIONS)
    qs = {a: Q[(s, a)] for a in ACTIONS}
    max_q = max(qs.values())
    greedy = [a for a, v in qs.items() if v == max_q]

    probs = {a: eps / nA for a in ACTIONS}
    greedy_mass = 1.0 - (nA - 1) * (eps / nA)

    for a in greedy:
        probs[a] += greedy_mass / len(greedy)

    return probs


def sample_from_probs(probs: Dict[str, float]):
    r = random.random()
    cumulative = 0.0
    for a, p in probs.items():
        cumulative += p
        if r <= cumulative:
            return a
    raise ValueError("Cummulative probability should be 1.0")


def select_action_epsilon_greedy(Q, s):
    probs = epsilon_greedy_probs(Q, s, EPS)
    return sample_from_probs(probs)
