# Implementation-of-Iterative-Policy-Evaluation-for-a-Finite-MDP

## Date : 27-10-2026

## Aim

To implement iterative policy evaluation using Gymnasium and estimate the state-value function $V^\pi(s)$ for a fixed random policy.

---
## Software Requirements

Install the required Python packages:

```bash
pip install gymnasium numpy
```

---

## Environment Used

The experiment uses the **FrozenLake-v1** environment from Gymnasium.

FrozenLake is a grid-based reinforcement learning environment where the agent starts from a start state and tries to reach the goal state without falling into holes.

For the default 4 x 4 FrozenLake map:

| Component | Description |
|---|---|
| Observation space | 16 discrete states |
| Action space | 4 discrete actions |
| Actions | 0 = Left, 1 = Down, 2 = Right, 3 = Up |
| Reward | +1 for reaching goal, 0 otherwise |
| Terminal states | Goal and holes |

---

## Problem Statement

Evaluate a fixed random policy in the FrozenLake-v1 environment.

The agent follows a random policy, where each of the four actions is selected with equal probability:

$$
\pi(a|s) = \frac{1}{4}
$$

This probability refers to the policy's action-selection probability. The environment transition probabilities are obtained from Gymnasium using `env.P[state][action]`. If `is_slippery=True`, the agent may not move in the intended direction due to stochastic transitions.

The objective is to estimate the state-value function:

$$
V^\pi(s)
$$

---

## Theory

The state-value function under policy $pi$, denoted by $V^\pi(s)$, represents the expected return starting from state $s$ and following policy $pi$.

The Bellman expectation equation is:

```math
V^\pi(s) =
\sum_a \pi(a|s)
\sum_{s'} P(s'|s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
```

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action |
| $s'$ | Next state |
| $\pi(a \mid s)$ | Probability of selecting action $a$ in state $s$ |
| $P(s' \mid s,a)$ | Transition probability |
| $R(s,a,s')$ | Reward |
| $\gamma$ | Discount factor |
| $V^\pi(s)$ | Value of state $s$ under policy $\pi$ |

---
## Algorithm

1. Create the FrozenLake-v1 environment using Gymnasium.
2. Access the transition model of the environment.
3. Initialize \(V(s)=0\) for all states.
4. Define a random policy where each action has equal probability.
5. For each state:
   - For each action:
     - Read transition probability, next state, reward, and terminal status.
     - Apply the Bellman expectation equation.
6. Repeat until the value function converges.
7. Display the final value function as a 4 x 4 grid.

---

## Program

```python
----------------------------
# Policy Evaluation Function
----------------------------
def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):
    V = np.zeros(env.observation_space.n)     # initial values are zero
    iteration = 0

    while True:
        delta = 0
        iteration += 1

        for s in range(env.observation_space.n):          # loop through all states
            v_new = 0
            for a in range(env.action_space.n):           # loop through all actions
                for prob, next_state, reward, done in env.P[s][a]:
                    v_new += policy[s][a] * prob * (reward + gamma * V[next_state] * (not done))

            delta = max(delta, abs(v_new - V[s]))         # track largest change
            V[s] = v_new                                  # update state value

        if delta < theta:                                 # check convergence
            break

    return V, iteration
```
```python
----------------------
# Run Policy Iteration
----------------------
V, iterations = policy_evaluation(env, policy, gamma, theta)

print("Name: Kishore S")
print("Register Number: 212224230130")
print("Number of iterations:", iterations)
print("\nState-Value Function:")
print(V)

print("Name: Kishore S")
print("Register Number: 212224230130")
print("\nState-Value Function as 4x4 Grid:")
print(np.round(V.reshape(4, 4), 4))

env.close()
```

---

## Output

#### For Gamma = 0.99

<img width="892" height="369" alt="image" src="https://github.com/user-attachments/assets/7f718de6-9a94-4415-b3e3-91f6d4f20758" />


#### For Gamma = 0.9

<img width="759" height="363" alt="image" src="https://github.com/user-attachments/assets/3f779eef-7003-4f03-989f-c96647b6a219" />

---

## Result

Iterative policy evaluation was implemented successfully using the Gymnasium FrozenLake environment. The state-value function for the fixed random policy was estimated using the Bellman expectation equation.

---

## Inference

- The state-value function Vπ(s) estimates how good each state is when the agent follows the fixed random policy (all four actions equally likely).
- The values are highest for states close to the goal. State 14, which is next to the goal, has the highest value (V(14) = 0.4336), followed by state 13 (0.1703) and state 10 (0.1378).
- The values decrease as the distance from the goal increases. The start state 0 has a very low value (V(0) = 0.0124), so a random agent rarely reaches the goal from the start.
- Hole states (5, 7, 11, 12) and the goal state (15) have value 0 because they are terminal states and no future reward can be collected from them.
- The overall values are small because the policy is random and the environment is slippery, so the agent often slips into holes or wanders away from the goal.
- The algorithm converged in 54 iterations, which means the value estimates became stable once the change in every state was smaller than θ = 1e-8.
- The discount factor γ = 0.99 makes rewards that are far away slightly less valuable, so states farther from the goal receive smaller values.
- Policy evaluation only measures how good a given policy is. It does not improve the policy, so the next step would be policy improvement or policy iteration to find a better policy.


