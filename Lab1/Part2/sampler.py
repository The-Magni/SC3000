import random

from utils import (
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
