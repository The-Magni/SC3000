from sampler import valid_next
from utils import (
    ACTIONS,
    GAMMA,
    GOAL,
    GOAL_REWARD,
    STATES,
    STEP_COST,
    V_table,
    compare_policies,
    is_terminal,
    policy_table,
)

# Stochastic transition model:
# 0.8 intended direction
# 0.1 perpendicular direction 1
# 0.1 perpendicular direction 2
PERP = {
    "U": ("L", "R"),
    "D": ("L", "R"),
    "L": ("U", "D"),
    "R": ("U", "D"),
}


def transition_probs(s, a):
    """
    Return a list of (probability, next_state, reward) outcomes for taking action a in state s.
    Uses valid_next() from sampler.py for movement handling.
    """
    if is_terminal(s):
        return [(1.0, s, 0.0)]

    outcomes = {}
    executed_actions = [
        (0.8, a),
        (0.1, PERP[a][0]),
        (0.1, PERP[a][1]),
    ]

    for p, executed_a in executed_actions:
        s_next = valid_next(s, executed_a)
        reward = GOAL_REWARD if s_next == GOAL else STEP_COST
        key = (s_next, reward)
        outcomes[key] = outcomes.get(key, 0.0) + p

    return [(p, s_next, reward) for (s_next, reward), p in outcomes.items()]


def expected_return(s, a, V):
    total = 0.0
    for p, s_next, reward in transition_probs(s, a):
        total += p * (reward + GAMMA * V[s_next])
    return total


def greedy_action_from_V(s, V):
    if is_terminal(s):
        return None
    return max(ACTIONS, key=lambda a: expected_return(s, a, V))


def value_iteration(theta=1e-10, max_iters=10000):
    V = {s: 0.0 for s in STATES}
    iter_idx = 1
    for iter_idx in range(1, max_iters + 1):
        delta = 0.0
        V_new = V.copy()

        for s in STATES:
            if is_terminal(s):
                V_new[s] = 0.0
                continue

            V_new[s] = max(expected_return(s, a, V) for a in ACTIONS)
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new

        if delta < theta:
            break

    pi = {s: greedy_action_from_V(s, V) for s in STATES}
    return V, pi, iter_idx


def policy_evaluation(pi, theta=1e-10, max_iters=10000):
    V = {s: 0.0 for s in STATES}

    for _ in range(max_iters):
        delta = 0.0
        V_new = V.copy()

        for s in STATES:
            if is_terminal(s):
                V_new[s] = 0.0
                continue

            V_new[s] = expected_return(s, pi[s], V)
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new

        if delta < theta:
            break

    return V


def policy_iteration(theta=1e-10, max_policy_iters=10000):
    pi = {}
    for s in STATES:
        if is_terminal(s):
            pi[s] = None
        else:
            pi[s] = "R"
    iter_idx = 1
    for iter_idx in range(1, max_policy_iters + 1):
        V = policy_evaluation(pi, theta=theta)

        policy_stable = True
        for s in STATES:
            if is_terminal(s):
                continue

            old_action = pi[s]
            pi[s] = greedy_action_from_V(s, V)

            if pi[s] != old_action:
                policy_stable = False

        if policy_stable:
            break

    V = policy_evaluation(pi, theta=theta)
    return V, pi, iter_idx


def main():
    print("=== Value Iteration ===")
    V_vi, pi_vi, vi_iters = value_iteration()
    print(f"Converged in {vi_iters} iterations")
    print(V_table(V_vi).round(4))
    print(policy_table(pi_vi))
    print()

    print("=== Policy Iteration ===")
    V_pi, pi_pi, pi_iters = policy_iteration()
    print(f"Converged in {pi_iters} policy improvement rounds")
    print(V_table(V_pi).round(4))
    print(policy_table(pi_pi))
    print()

    mismatches = compare_policies(pi_vi, pi_pi)
    if len(mismatches) == 0:
        print("Value iteration and policy iteration produced the same optimal policy.")
    else:
        print("Policies differ at the following states:")
        for s, a_vi, a_pi in mismatches:
            print(f"{s}: VI={a_vi}, PI={a_pi}")


if __name__ == "__main__":
    main()
