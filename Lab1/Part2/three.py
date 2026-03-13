import random

import matplotlib.pyplot as plt
from one import value_iteration
from sampler import EPS, sample_from_probs, step_stochastic_sample
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


def q_learning(alpha: float = 0.1, num_episodes: int = 50000, seed: int = 0):
    random.seed(seed)
    # randomly generate Q
    Q = {(s, a): random.random() for s in STATES for a in ACTIONS}
    avg_deltas = []
    for epoch in range(num_episodes):
        s = START
        deltas = []
        while not is_terminal(s):
            policy_s = {a: 0.0 for a in ACTIONS}
            a_star = max(ACTIONS, key=lambda a: Q[(s, a)])
            for a in ACTIONS:
                if a == a_star:
                    policy_s[a] = 1.0 - EPS + EPS / len(ACTIONS)
                else:
                    policy_s[a] = EPS / len(ACTIONS)
            a = sample_from_probs(policy_s)
            s_next, reward, _ = step_stochastic_sample(s, a)
            if is_terminal(s_next):
                V_s_next = 0.0
            else:
                V_s_next = max(Q[(s_next, a)] for a in ACTIONS)
            deltas.append(alpha * (reward + GAMMA * V_s_next - Q[(s, a)]))
            Q[(s, a)] = Q[(s, a)] + alpha * (reward + GAMMA * V_s_next - Q[(s, a)])
            s = s_next
        avg_deltas.append(sum(deltas) / len(deltas))
    return Q, avg_deltas


def main():
    # Optimal policy from Task 1
    V_opt, pi_opt, _ = value_iteration()
    num_episodes = 1000
    # Learned policy from Task 3
    Q, avg_deltas = q_learning(num_episodes=num_episodes)
    pi_q = greedy_policy_from_Q(Q)
    V_q = V_from_Q(Q)
    print(V_table(V_q))
    print("=== Q-Learning Policy ===")
    print(policy_table(pi_q))
    print()

    # plotting
    plt.plot(range(num_episodes), avg_deltas)
    plt.ylim(-2, 2)
    plt.xlabel("Epoch")
    plt.ylabel("Avergage change in Q values")
    plt.show()
    print("=== Optimal Policy from Task 1 ===")
    print(policy_table(pi_opt))
    print()

    matches, total, accuracy, mismatches = compare_policy_against_optimal(pi_opt, pi_q)

    print(f"Policy agreement: {matches}/{total} = {accuracy:.2%}")

    if len(mismatches) == 0:
        print("Q-learning policy matches the optimal policy exactly.")
    else:
        print("States with different actions:")
        for s, a_opt, a_q in mismatches:
            print(f"{s}: optimal={a_opt}, q_learning={a_q}")


if __name__ == "__main__":
    main()
