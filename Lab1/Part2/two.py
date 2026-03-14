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
    compare_policy_against_optimal,
    greedy_policy_from_Q,
    is_terminal,
    policy_table,
)


def monte_carlo_learning(num_episodes: int = 10000, seed: int = 0):
    random.seed(seed)
    Q = {(s, a): random.random() for s in STATES for a in ACTIONS}
    Returns = defaultdict(list)
    policy = {s: {a: 1.0 / len(ACTIONS) for a in ACTIONS} for s in STATES}
    # randomly generated policy
    avg_deltas = []
    for epoch in range(num_episodes):
        visited = set()
        episode = []
        s = START
        deltas = []
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
            Q_old = Q[(s, a)]
            Q[(s, a)] = sum(Returns[(s, a)]) / len(Returns[(s, a)])
            deltas.append(Q[(s, a)] - Q_old)
        visited_states = {s for s, _ in visited}
        for s in visited_states:
            a_star = max(ACTIONS, key=lambda a: Q[(s, a)])
            for a in ACTIONS:
                if a == a_star:
                    policy[s][a] = 1.0 - EPS + EPS / len(ACTIONS)
                else:
                    policy[s][a] = EPS / len(ACTIONS)
        avg_deltas.append(sum(deltas) / len(deltas))
        # progress logging removed from inner loop (only final result printed later)
    # after all episodes, display final iteration results
    return Q, avg_deltas


SEP = "=" * 50


def main():
    V_star, pi_star, _ = value_iteration()
    num_episodes = 10000
    Q, _ = monte_carlo_learning(num_episodes=num_episodes)
    pi = greedy_policy_from_Q(Q)
    V = V_from_Q(Q)

    print(f"\n{SEP}")
    print("  Value Table (Monte Carlo)")
    print(SEP)
    print(V_table(V))

    print(f"\n{SEP}")
    print("  Monte Carlo Policy")
    print(SEP)
    print(policy_table(pi))

    print(f"\n{SEP}")
    print("  Optimal Policy (Task 1 - Value Iteration)")
    print(SEP)
    print(policy_table(pi_star))

    print(f"\n{SEP}")
    print("  Policy Comparison")
    print(SEP)
    matches, total, accuracy, mismatches = compare_policy_against_optimal(pi_star, pi)
    print(f"  Policy agreement : {matches} / {total}  ({accuracy:.2%})")

    if len(mismatches) == 0:
        print("  [MATCH] Monte Carlo policy matches the optimal policy exactly.")
    else:
        print(f"  [DIFF] Diverging states ({len(mismatches)}):")
        for s, a_opt, a_mc in mismatches:
            print(f"     - State {str(s):<10} | optimal={a_opt:<6} | monte_carlo={a_mc}")
    print(SEP)


if __name__ == "__main__":
    main()
