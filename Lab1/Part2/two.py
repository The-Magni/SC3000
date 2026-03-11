import random
from collections import defaultdict

from one import value_iteration
from sampler import (
    EPS,
    sample_from_probs,
    step_stochastic_sample,
)
from utils import (
    ACTIONS,
    GAMMA,
    START,
    STATES,
    V_from_Q,
    V_table,
    greedy_policy_from_Q,
    is_terminal,
    policy_table,
    compare_policy_against_optimal
)


def monte_carlo_learning(num_episodes: int = 10000, seed: int = 0):
    random.seed(seed)
    Q = {(s, a): random.random() * 1000 for s in STATES for a in ACTIONS}
    Returns = defaultdict(list)
    policy = {s: {a: 1.0 / len(ACTIONS) for a in ACTIONS} for s in STATES}
    # randomly generated policy
    for iter in range(num_episodes):
        visited = set()
        episode = []
        s = START
        while not is_terminal(s):
            a = sample_from_probs(policy[s])
            s_next, reward, _ = step_stochastic_sample(s, a)
            episode.append((s, a, reward))
            s = s_next

        for i, (s, a, _) in enumerate(episode):
            if (s, a) in visited:
                continue
            visited.add((s, a))
            ret = 0
            multiplier = 1
            for j in range(i, len(episode)):
                _, _, reward = episode[j]
                ret += multiplier * reward
                multiplier *= GAMMA
            Returns[(s, a)].append(ret)
            Q[(s, a)] = sum(Returns[(s, a)]) / len(Returns[(s, a)])
        visited_states = {s for s, _ in visited}
        for s in visited_states:
            a_star = ACTIONS[0]
            for a in ACTIONS[1:]:
                if Q[(s, a)] > Q[(s, a_star)]:
                    a_star = a
            for a in ACTIONS:
                if a == a_star:
                    policy[s][a] = 1.0 - EPS + EPS / len(ACTIONS)
                else:
                    policy[s][a] = EPS / len(ACTIONS)

        # progress logging removed from inner loop (only final result printed later)
    # after all episodes, display final iteration results
    print(f"Final value iteration of Monte Carlo Policy:")
    print(V_table(V_from_Q(Q)))
    print()
    return Q


if __name__ == "__main__":
    V_star, pi_star, _ = value_iteration()

    Q = monte_carlo_learning(num_episodes=10000)
    pi = greedy_policy_from_Q(Q)
    V = V_from_Q(Q)

    print("=== Monte Carlo Policy ===")
    print(policy_table(pi))
    print()

    print("=== Optimal Policy from Task 1 ===")
    print(policy_table(pi_star))
    print()

    matches, total, accuracy, mismatches = compare_policy_against_optimal(pi_star, pi)

    print(f"Policy agreement: {matches}/{total} = {accuracy:.2%}")

    if len(mismatches) == 0:
        print("Monte Carlo policy matches the optimal policy exactly.")
    else:
        print("States with different actions:")
        for s, a_opt, a_mc in mismatches:
            print(f"{s}: optimal={a_opt}, monte_carlo={a_mc}")
