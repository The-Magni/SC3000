import random

from sampler import EPS, sample_from_probs, step_stochastic_sample
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
)


def q_learning(alpha: float = 0.1, num_episodes: int = 50000, seed: int = 0):
    random.seed(seed)
    # randomly generate Q
    Q = {(s, a): random.random() * 1000 for s in STATES for a in ACTIONS}
    for iter in range(num_episodes):
        s = START
        while not is_terminal(s):
            policy_s = {a: 0.0 for a in ACTIONS}
            a_star = ACTIONS[0]
            for a in ACTIONS[1:]:
                if Q[(s, a)] > Q[(s, a_star)]:
                    a_star = a

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
            Q[(s, a)] = Q[(s, a)] + alpha * (reward + GAMMA * V_s_next - Q[(s, a)])
            s = s_next
        print(f"iter {iter}")
        print(V_table(V_from_Q(Q)))
    return Q


if __name__ == "__main__":
    Q = q_learning(num_episodes=10000)
    pi_star = greedy_policy_from_Q(Q)
    V = V_from_Q(Q)
    print(V_table(V))
    print(policy_table(pi_star))
