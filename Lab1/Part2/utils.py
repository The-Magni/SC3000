import pandas as pd

# Grid settings
N = 5
START = (0, 0)
GOAL = (4, 4)
BLOCKS = {(1, 2), (3, 2)}
STATES = [(x, y) for x in range(N) for y in range(N) if (x, y) not in BLOCKS]
GAMMA = 0.9

# Actions and movement
ACTIONS = ["U", "D", "L", "R"]
DELTA = {"U": (0, 1), "D": (0, -1), "L": (-1, 0), "R": (1, 0)}

# MDP settings
STEP_COST = -1
GOAL_REWARD = 10
ARROW = {"U": "↑", "D": "↓", "L": "←", "R": "→", None: "·"}

def compare_policy_against_optimal(pi_opt, pi_test):
    matches = 0
    total = 0
    mismatches = []

    for s in STATES:
        if is_terminal(s):
            continue

        total += 1
        if pi_opt[s] == pi_test[s]:
            matches += 1
        else:
            mismatches.append((s, pi_opt[s], pi_test[s]))

    accuracy = matches / total
    return matches, total, accuracy, mismatches

def greedy_policy_from_Q(Q):
    pi = {}
    for s in STATES:
        if is_terminal(s):
            pi[s] = None
        else:
            pi[s] = max(ACTIONS, key=lambda a: Q[(s, a)])
    return pi


def V_from_Q(Q):
    V = {}
    for s in STATES:
        if is_terminal(s):
            V[s] = 0.0
        else:
            V[s] = max(Q[(s, a)] for a in ACTIONS)
    return V


def V_table(V):
    """
    Returns a 5x5 DataFrame indexed by y (top to bottom) and columns x (left to right).
    """
    grid = []
    for y in reversed(range(N)):
        row = []
        for x in range(N):
            s = (x, y)
            if s in BLOCKS:
                row.append(None)
            elif s == GOAL:
                row.append(0.0)
            else:
                row.append(V.get(s, None))
        grid.append(row)
    df = pd.DataFrame(
        grid, columns=pd.Index(range(N)), index=pd.Index(reversed(range(N)))
    )
    return df


def policy_table(pi):
    """
    Returns a 5x5 DataFrame of arrow symbols.
    """
    grid = []
    for y in reversed(range(N)):
        row = []
        for x in range(N):
            s = (x, y)
            if s in BLOCKS:
                row.append("####")
            elif s == GOAL:
                row.append("GOAL")
            else:
                row.append(ARROW[pi.get(s, None)])
        grid.append(row)
    df = pd.DataFrame(
        grid, columns=pd.Index(range(N)), index=pd.Index(reversed(range(N)))
    )
    return df


def Q_table(Q):
    """
    Returns a DataFrame with index=states and columns=actions.
    """
    rows = []
    for s in STATES:
        if s in BLOCKS:
            continue
        row = {"state": s}
        for a in ACTIONS:
            row[a] = Q.get((s, a), 0.0)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("state").sort_index()
    return df


def is_terminal(s):
    return s == GOAL


def in_bounds(cell):
    x, y = cell
    return 0 <= x < N and 0 <= y < N


def is_blocked(cell):
    return cell in BLOCKS
