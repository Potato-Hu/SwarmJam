# System Model

## 1. Overview

We consider a multi-UAV cooperative jamming scenario against an adversarial swarm consisting of key and non-key nodes. The goal of the friendly UAV swarm is to **selectively suppress key nodes** while minimizing interference to non-key nodes and avoiding excessive mutual interference.

The system integrates:
- Ground-truth simulation of enemy swarm
- Noisy global sensing
- Local onboard sensing
- A simplified JPDA-based association module
- A cooperative MARL (MAPPO) control policy

---

## 2. Entities

### 2.1 Friendly UAVs
- Number: $N_f$
- Equipped with omnidirectional antennas
- Action space:
  - Continuous velocity control
  - Optional power allocation (if modeled)

State of UAV $i$:
$$
 s_i = (x_i, v_i)
$$

---

### 2.2 Enemy Nodes

Two types:

- **Key nodes** (critical targets)
- **Non-key nodes** (clutter / decoys)

Total number: $N_e = N_k + N_c$

State of node $j$:
$$
 x_j = (p_j, v_j)
$$

---

## 3. Dynamics Model

All agents evolve in discrete time:
$$
 x(t+1) = f(x(t)) + w(t)
$$

- $f(\cdot)$: motion model (e.g., constant velocity)
- $w(t)$: process noise

---

## 4. Sensing Model

### 4.1 Global Sensor

Provides noisy observations of (mainly) key nodes:
$$
 z_k^{global}(t) = p_k(t) + \epsilon_k(t)
$$

- $\epsilon_k \sim \mathcal{N}(0, \Sigma_g)$
- May include missed detections or delay

---

### 4.2 Local Sensor (per UAV)

Each UAV observes nearby targets:

$$
 z_{i,j}^{local}(t) = p_j(t) + \eta_{i,j}(t)
$$

- Limited sensing range
- No explicit key/non-key label

---

## 5. Association Model (Simplified JPDA)

We define a soft association score between global observation $k$ and local observation $j$:

$$
 S_{k,j}(t) = \exp(-\|z_k^{global} - z_{i,j}^{local}\|^2 / \sigma^2)
$$

Normalized via softmax:

$$
 P_{k,j}(t) = \frac{S_{k,j}}{\sum_j S_{k,j}}
$$

Then construct **key target estimates**:

$$
 \hat{x}_k(t) = \sum_j P_{k,j}(t) \cdot z_{i,j}^{local}(t)
$$

---

## 6. Channel & Interference Model

Received interference power from UAV $i$ to node $j$:

$$
 I_{i,j}(t) = P_i \cdot G(d_{i,j}) \cdot H_{i,j}
$$

Where:
- $G(d)$: path-loss
- $H$: Rician fading

Total received power:

$$
 I_j(t) = \sum_i I_{i,j}(t)
$$

---

## 7. Reward-Relevant Quantities

- Key node gain: increasing $I_k$
- Non-key penalty: suppressing wrong targets
- Friendly interference penalty

---
