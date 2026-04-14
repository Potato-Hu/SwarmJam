# Problem Formulation

## 1. Objective

We formulate the task as a cooperative multi-agent reinforcement learning problem.

Goal:

> Maximize long-term suppression effectiveness on key nodes under sensing uncertainty and resource constraints.

---

## 2. State, Observation, Action

### 2.1 Global State (for critic)

Includes:
- All UAV states
- All (true or estimated) enemy states

---

### 2.2 Agent Observation

Each UAV observes:

- Self state: $(x_i, v_i)$
- Local observations
- Estimated key targets $\hat{x}_k$

---

### 2.3 Action Space

For UAV $i$:

$$
 a_i = (u_i)
$$

Where $u_i$ is velocity (and optionally power).

---

## 3. Reward Function

We define the reward as:

$$
 r(t) = R_{key}(t) - R_{nonkey}(t) - R_{friendly}(t)
$$

### 3.1 Key Gain

$$
 R_{key} = \sum_{k \in key} w_k \cdot I_k
$$

---

### 3.2 Non-Key Penalty

$$
 R_{nonkey} = \sum_{j \in nonkey} I_j
$$

---

### 3.3 Friendly Interference Penalty

$$
 R_{friendly} = \lambda \cdot \sum_i \mathbb{1}(I_i^{self} > threshold)
$$

---

## 4. Policy Optimization

We adopt MAPPO:

- Centralized critic
- Decentralized actors

Objective:

$$
 \max_\theta \mathbb{E}[\sum_t \gamma^t r(t)]
$$

---

## 5. Key Challenges

1. Partial observability
2. Data association uncertainty
3. Multi-agent coordination
4. Trade-off between exploitation and interference control

---

## 6. Contribution (Aligned with Model)

- Introduce simplified JPDA for key node inference
- Integrate sensing uncertainty into MARL
- Joint optimization of motion and interference

