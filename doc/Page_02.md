Awesome—let’s extend your **single real-world exercise (e-bike dynamic pricing)** in two beginner-friendly ways:

1. **Add CSV logging + clean Matplotlib plots** to your tabular Q-Learning script.
2. **A tiny DQN (Deep Q-Network)** version that uses a small neural net, replay buffer, and a target network—kept minimal for macOS/CPU.

You can do either or both.

---

# 0) Environment prerequisites (macOS, CPU)

```bash
# If not already created
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# For both scripts
pip install numpy matplotlib

# For DQN (PyTorch CPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu
# (If that index fails on your setup, fall back to: pip install torch)
```

---

# 1) Tabular Q-Learning + CSV Logging + Plots

Save as **`bike_pricing_qlearning_logged.py`** (a drop-in replacement for your earlier file).
What’s new:

* Logs per-episode train returns to **`training_log.csv`**.
* Adds **evaluation runs**, logs to CSV.
* Produces two **Matplotlib** charts (no seaborn): moving average training curve and evaluation curve.

```python
# bike_pricing_qlearning_logged.py
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# -------------------------
# Config & Hyperparameters
# -------------------------
SLOTS_PER_DAY = 96
BIKE_CAPACITY = 40
INV_BINS = BIKE_CAPACITY + 1
PRICE_VALUES = {0: 1.0, 1: 1.5, 2: 2.0}
PRICE_SENSITIVITY = 0.55
BASE_DEMAND = 3.0
PEAK_BONUS_AM = 2.5
PEAK_BONUS_PM = 2.0
NOISE_SD = 0.8
STOCKOUT_PENALTY = 1.0
LEFTOVER_PENALTY = 0.05

ALPHA = 0.25
GAMMA = 0.98
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9995
EPISODES = 2500
EVAL_EVERY = 100
EVAL_EPISODES = 10
LOG_CSV = "training_log.csv"
rng = np.random.default_rng(42)

@dataclass
class StepResult:
    next_state: tuple
    reward: float
    done: bool
    info: dict

class BikePricingEnv:
    def __init__(self):
        self.capacity = BIKE_CAPACITY
        self.t = 0
        self.inv = self.capacity

    def reset(self):
        self.t = 0
        self.inv = self.capacity
        return (self.t, self.inv)

    def demand_mean(self, t_slot, price_tier):
        hour = (t_slot * 15) / 60.0
        mean = BASE_DEMAND
        if 8 <= hour < 10: mean += PEAK_BONUS_AM
        if 17 <= hour < 19: mean += PEAK_BONUS_PM
        mean *= np.exp(-PRICE_SENSITIVITY * price_tier)
        return max(0.0, mean)

    def step(self, action):
        price_tier = int(action)
        mean = self.demand_mean(self.t, price_tier)
        demand = max(0.0, rng.normal(loc=mean, scale=NOISE_SD))
        rentals = int(min(self.inv, np.floor(demand)))
        turned_away = max(0, int(np.floor(demand)) - rentals)

        revenue = rentals * PRICE_VALUES[price_tier]
        penalty = turned_away * STOCKOUT_PENALTY
        reward = revenue - penalty

        self.inv -= rentals
        self.t += 1
        done = self.t >= SLOTS_PER_DAY
        info = {"rentals": rentals, "turned_away": turned_away, "revenue": revenue}

        if done and self.inv > 0:
            leftover_pen = LEFTOVER_PENALTY * self.inv
            reward -= leftover_pen
            info["leftover_penalty"] = leftover_pen

        next_state = (min(self.t, SLOTS_PER_DAY - 1), self.inv) if not done else (SLOTS_PER_DAY - 1, self.inv)
        return StepResult(next_state, reward, done, info)

class QLearningAgent:
    def __init__(self, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON_START):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((SLOTS_PER_DAY, INV_BINS, 3), dtype=np.float32)

    def select_action(self, state, greedy=False):
        t, inv = state
        if (not greedy) and (rng.random() < self.epsilon):
            return int(rng.integers(0, 3))
        return int(np.argmax(self.Q[t, inv, :]))

    def update(self, s, a, r, s_next):
        t, inv = s
        t2, inv2 = s_next
        best_next = np.max(self.Q[t2, inv2, :])
        td_target = r + self.gamma * best_next
        self.Q[t, inv, a] += self.alpha * (td_target - self.Q[t, inv, a])

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

def run_episode(env, agent, greedy=False):
    s = env.reset()
    total_r = 0.0
    for _ in range(SLOTS_PER_DAY):
        a = agent.select_action(s, greedy=greedy)
        step = env.step(a)
        if not greedy:
            agent.update(s, a, step.reward, step.next_state)
        total_r += step.reward
        s = step.next_state
        if step.done:
            break
    return total_r

def evaluate(env, agent, episodes=5):
    return float(np.mean([run_episode(env, agent, greedy=True) for _ in range(episodes)]))

def main():
    env = BikePricingEnv()
    agent = QLearningAgent()

    # CSV header
    with open(LOG_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "train_return", "eval_return", "epsilon"])

    train_returns = []
    eval_points_x, eval_points_y = [], []

    for ep in range(1, EPISODES + 1):
        tr = run_episode(env, agent, greedy=False)
        agent.decay_epsilon()
        train_returns.append(tr)

        eval_r = ""
        if ep % EVAL_EVERY == 0:
            eval_r = evaluate(env, agent, episodes=EVAL_EPISODES)
            eval_points_x.append(ep)
            eval_points_y.append(eval_r)
            print(f"Episode {ep:4d} | TrainAvg(last {EVAL_EVERY})={np.mean(train_returns[-EVAL_EVERY:]):.2f} "
                  f"| Eval={eval_r:.2f} | ε={agent.epsilon:.3f}")

        with open(LOG_CSV, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([ep, f"{tr:.4f}", f"{eval_r}" if eval_r != "" else "", f"{agent.epsilon:.6f}"])

    # Plots
    plt.figure()
    plt.plot(train_returns, alpha=0.35, label="Episode return")
    if len(train_returns) >= 50:
        win = 50
        moving = np.convolve(train_returns, np.ones(win) / win, mode="valid")
        plt.plot(range(win - 1, len(train_returns)), moving, label=f"Moving avg ({win})")
    if eval_points_x:
        plt.plot(eval_points_x, eval_points_y, marker="o", label="Evaluation (greedy)")
    plt.title("Q-Learning: Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Quick peek: greedy policy at 8:00 by inventory
    t = 8 * 4
    greedy_actions = np.argmax(agent.Q[t, :, :], axis=1)
    names = {0: "Low", 1: "Med", 2: "High"}
    print("\nGreedy price policy at 08:00 by inventory:")
    for inv in range(0, BIKE_CAPACITY + 1, 5):
        print(f"  inv={inv:2d} -> {names[int(greedy_actions[inv])]}")

if __name__ == "__main__":
    main()
```

### What this teaches (beyond basics)

* **Experiment tracking**: inspect `training_log.csv` to compare runs/hyperparameters.
* **Evaluation discipline**: separates noisy training returns from greedy evaluation.

---

# 2) Tiny DQN (Deep Q-Network) for the Same Problem

Why DQN here:

* Lets you **scale** to larger state spaces later (e.g., add weather, weekday/weekend, recent demand).
* Teaches **replay buffer**, **target network**, **stability tricks** (Huber loss, gradient clipping).

### Key design choices (kept simple)

* **State** = `[one-hot time(96 dims), inventory/40]` → length 97; still simple but NN-friendly.
* **MLP**: 2 hidden layers (128, 128), ReLU.
* **Replay buffer** with uniform sampling.
* **Target network** update every N steps.
* **Epsilon decay** like tabular case.
* **Multiple gradient steps per environment step** kept at 1 for clarity (you can increase later).

Save as **`bike_pricing_dqn.py`**:

```python
# bike_pricing_dqn.py
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from collections import deque

# -------------------------
# Env config (same logic as tabular)
# -------------------------
SLOTS_PER_DAY = 96
BIKE_CAPACITY = 40
PRICE_VALUES = {0: 1.0, 1: 1.5, 2: 2.0}
PRICE_SENSITIVITY = 0.55
BASE_DEMAND = 3.0
PEAK_BONUS_AM = 2.5
PEAK_BONUS_PM = 2.0
NOISE_SD = 0.8
STOCKOUT_PENALTY = 1.0
LEFTOVER_PENALTY = 0.05

# -------------------------
# DQN Hyperparameters
# -------------------------
GAMMA = 0.98
LR = 1e-3
BATCH_SIZE = 64
REPLAY_CAP = 50_000
TRAIN_START = 1_000          # start learning after this many transitions
TARGET_UPDATE_EVERY = 1_000  # steps
EPISODES = 2000
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9995
GRAD_CLIP_NORM = 5.0
DEVICE = torch.device("cpu")  # keep CPU for portability

rng = np.random.default_rng(1234)

@dataclass
class StepResult:
    next_state: np.ndarray
    reward: float
    done: bool
    info: dict

class BikePricingEnv:
    def __init__(self):
        self.capacity = BIKE_CAPACITY
        self.t = 0
        self.inv = self.capacity

    def reset(self):
        self.t = 0
        self.inv = self.capacity
        return self._obs()

    def demand_mean(self, t_slot, price_tier):
        hour = (t_slot * 15) / 60.0
        mean = BASE_DEMAND
        if 8 <= hour < 10: mean += PEAK_BONUS_AM
        if 17 <= hour < 19: mean += PEAK_BONUS_PM
        mean *= math.exp(-PRICE_SENSITIVITY * price_tier)
        return max(0.0, mean)

    def _obs(self):
        # state = [one-hot time(96), inv_norm]
        time_oh = np.zeros(SLOTS_PER_DAY, dtype=np.float32)
        time_oh[self.t] = 1.0
        inv_norm = np.array([self.inv / float(self.capacity)], dtype=np.float32)
        return np.concatenate([time_oh, inv_norm], axis=0)  # shape (97,)

    def step(self, action):
        price_tier = int(action)
        mean = self.demand_mean(self.t, price_tier)
        demand = max(0.0, rng.normal(loc=mean, scale=NOISE_SD))
        rentals = int(min(self.inv, math.floor(demand)))
        turned_away = max(0, int(math.floor(demand)) - rentals)

        revenue = rentals * PRICE_VALUES[price_tier]
        penalty = turned_away * STOCKOUT_PENALTY
        reward = revenue - penalty

        self.inv -= rentals
        self.t += 1
        done = self.t >= SLOTS_PER_DAY

        if done and self.inv > 0:
            reward -= LEFTOVER_PENALTY * self.inv

        return StepResult(self._obs() if not done else self._obs(), reward, done, {
            "rentals": rentals, "turned_away": turned_away, "revenue": revenue
        })

# -------------------------
# Replay Buffer
# -------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size):
        idx = rng.choice(len(self.buf), size=batch_size, replace=False)
        s, a, r, s2, d = zip(*(self.buf[i] for i in idx))
        return (np.stack(s), np.array(a), np.array(r, dtype=np.float32),
                np.stack(s2), np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buf)

# -------------------------
# Q-Network
# -------------------------
class QNet(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)

# -------------------------
# DQN Agent
# -------------------------
class DQNAgent:
    def __init__(self, state_dim, n_actions):
        self.n_actions = n_actions
        self.q = QNet(state_dim, n_actions).to(DEVICE)
        self.target = QNet(state_dim, n_actions).to(DEVICE)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=LR)
        self.loss_fn = nn.SmoothL1Loss()  # Huber
        self.epsilon = EPSILON_START

    def select_action(self, state, greedy=False):
        if (not greedy) and (rng.random() < self.epsilon):
            return int(rng.integers(0, self.n_actions))
        with torch.no_grad():
            s = torch.from_numpy(state).float().to(DEVICE).unsqueeze(0)
            qvals = self.q(s)
            return int(torch.argmax(qvals, dim=1).item())

    def train_step(self, batch):
        s, a, r, s2, d = batch
        s   = torch.from_numpy(s).float().to(DEVICE)
        a   = torch.from_numpy(a).long().to(DEVICE).unsqueeze(1)
        r   = torch.from_numpy(r).float().to(DEVICE).unsqueeze(1)
        s2  = torch.from_numpy(s2).float().to(DEVICE)
        d   = torch.from_numpy(d).float().to(DEVICE).unsqueeze(1)

        q_sa = self.q(s).gather(1, a)
        with torch.no_grad():
            max_q_s2 = self.target(s2).max(dim=1, keepdim=True).values
            target = r + (1.0 - d) * GAMMA * max_q_s2

        loss = self.loss_fn(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), GRAD_CLIP_NORM)
        self.opt.step()
        return float(loss.item())

    def update_target(self):
        self.target.load_state_dict(self.q.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

# -------------------------
# Training Loop
# -------------------------
def run_episode(env, agent, replay, total_steps, train=True):
    s = env.reset()
    ep_ret, ep_loss = 0.0, []

    for _ in range(SLOTS_PER_DAY):
        a = agent.select_action(s, greedy=not train)
        step = env.step(a)
        s2 = step.next_state
        replay.push(s, a, step.reward, s2, float(step.done))
        s = s2
        ep_ret += step.reward
        total_steps += 1

        if train and len(replay) >= TRAIN_START:
            batch = replay.sample(BATCH_SIZE)
            loss = agent.train_step(batch)
            ep_loss.append(loss)
            if total_steps % TARGET_UPDATE_EVERY == 0:
                agent.update_target()

        if step.done:
            break

    if train:
        agent.decay_epsilon()
    return ep_ret, (np.mean(ep_loss) if ep_loss else 0.0), total_steps

def evaluate(env, agent, episodes=5):
    returns = []
    for _ in range(episodes):
        s = env.reset()
        ep_ret = 0.0
        for _ in range(SLOTS_PER_DAY):
            a = agent.select_action(s, greedy=True)
            sr = env.step(a)
            s = sr.next_state
            ep_ret += sr.reward
            if sr.done:
                break
        returns.append(ep_ret)
    return float(np.mean(returns))

def main():
    env = BikePricingEnv()
    state_dim = SLOTS_PER_DAY + 1  # 96 one-hot + 1 inventory fraction
    n_actions = 3

    agent = DQNAgent(state_dim, n_actions)
    replay = ReplayBuffer(REPLAY_CAP)

    train_returns, eval_x, eval_y, losses = [], [], [], []
    total_steps = 0

    for ep in range(1, EPISODES + 1):
        ep_ret, ep_loss, total_steps = run_episode(env, agent, replay, total_steps, train=True)
        train_returns.append(ep_ret)
        losses.append(ep_loss)

        if ep % 100 == 0:
            eval_ret = evaluate(env, agent, episodes=10)
            eval_x.append(ep)
            eval_y.append(eval_ret)
            print(f"Ep {ep:4d} | TrainAvg(100)={np.mean(train_returns[-100:]):.2f} "
                  f"| Eval={eval_ret:.2f} | Loss={np.mean(losses[-100:]):.4f} | ε={agent.epsilon:.3f}")

    # Plots
    plt.figure()
    plt.plot(train_returns, alpha=0.35, label="Episode return")
    if len(train_returns) >= 50:
        win = 50
        mov = np.convolve(train_returns, np.ones(win)/win, mode="valid")
        plt.plot(range(win-1, len(train_returns)), mov, label=f"Moving avg ({win})")
    if eval_x:
        plt.plot(eval_x, eval_y, marker="o", label="Evaluation (greedy)")
    plt.title("DQN: Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Optional: visualize Q-values for one hour by inventory fraction
    # In DQN we don't have a table, but we can probe states.
    with torch.no_grad():
        hour8 = 8 * 4
        vals = []
        for inv in range(0, BIKE_CAPACITY + 1, 5):
            time_oh = np.zeros(SLOTS_PER_DAY, dtype=np.float32)
            time_oh[hour8] = 1.0
            inv_norm = np.array([inv / float(BIKE_CAPACITY)], dtype=np.float32)
            s = np.concatenate([time_oh, inv_norm], axis=0)
            q = agent.q(torch.from_numpy(s).float().unsqueeze(0)).squeeze(0).numpy()
            vals.append(q)
        vals = np.array(vals)  # shape (#inv, n_actions)
        plt.figure()
        for a in range(n_actions):
            plt.plot(range(0, BIKE_CAPACITY + 1, 5), vals[:, a], label=f"Q(action={a})")
        plt.title("Probed Q-values at 08:00 vs inventory")
        plt.xlabel("Inventory")
        plt.ylabel("Q")
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
```

### How DQN connects to RL concepts (hands-on)

* **Function approximation**: NN predicts Q(s, a) instead of a table.
* **Replay buffer**: breaks correlation in on-policy trajectories; stabilizes learning.
* **Target network**: fixed (slow-moving) bootstrapping target → avoids chasing a moving target.
* **Loss**: Huber (SmoothL1) is robust to outliers.
* **Gradient clipping**: prevents exploding updates.
* **Epsilon-greedy**: same exploration principle as tabular.

---

## How to run each

```bash
# Q-Learning with CSV logging
python bike_pricing_qlearning_logged.py
# -> training_log.csv + 2 plots

# DQN (CPU)
python bike_pricing_dqn.py
# -> progress prints + 2 plots
```

---

## Suggested experiments (to really “get” RL)

* **Epsilon schedule** (both): slower vs faster decay; notice exploration effects.
* **Stockout penalty** (both): raise to 2–3 → DQN/QL will price conservatively at low inventory.
* **Leftover penalty** (both): set to 0 → “sell out” behavior late in the day.
* **Network capacity** (DQN): try hidden sizes 64 or 256; watch stability/speed trade-offs.
* **Target update** (DQN): change `TARGET_UPDATE_EVERY` to 2k or 5k; see impact on smoothness.

---

## What you’re learning (mapping to fundamentals)

* **MDP framing** (state, action, reward, transitions).
* **TD bootstrapping** (both Q-Learning + DQN).
* **Off-policy control** (both aim for greedy policy while exploring).
* **Stability tools** (replay, target net, Huber, grad clip) that make deep RL practical.

If you’d like, I can also add:

* A **CSV logger** for DQN (mirroring Q-learning’s),
* A **checkpoint saver/loader** for the DQN model,
* A **hyperparameter sweep script** to compare ε-decays or penalties.
