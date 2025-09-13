Great choice. Here’s **one practical, real-world RL exercise in Python** you can run on macOS: **dynamic pricing for a shared e-bike station**. It’s realistic (micromobility companies do this), yet simple enough for a first RL project. You’ll build a simulator and a **tabular Q-Learning** agent that learns price policies to maximize daily revenue while avoiding stockouts.

# What you’ll learn (core RL concepts)

* **MDP pieces**: state (time + bikes left), action (price tier), reward (revenue – penalties), transitions (sales reduce inventory).
* **Exploration vs exploitation** with ε-greedy.
* **Temporal-Difference (TD) learning** and the **Q-Learning update**.
* **Discount factor (γ)** and how it trades off short-term vs long-term reward.
* **Reward shaping** (penalties for stockouts and leftover bikes).
* **Convergence intuition** via learning curves and greedy policy checks.

---

# Exercise: Dynamic Pricing for a Single E-Bike Station

## Problem

You manage a 40-bike station. Every **15 minutes** (96 steps/day), you set a **price tier** for rides (e.g., Low/Medium/High). Demand depends on **time-of-day** (commute peaks), **base popularity**, and the **chosen price** (higher price → fewer rentals). You earn revenue for each rental and incur:

* **Stockout penalty** if demand exceeds available bikes (lost goodwill).
* **Leftover penalty** at day end for idle bikes (under-utilization).

Goal: learn a pricing policy that **maximizes total daily reward**.

---

## macOS setup (Python)

```bash
# 1) Create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2) Install minimal deps
python -m pip install --upgrade pip
pip install numpy matplotlib
```

---

## How to run

Save the code below as `bike_pricing_qlearning.py`, then:

```bash
python bike_pricing_qlearning.py
```

It will train, print progress, and open two plots: average episode return and learned value heatmaps.

---

## Code (single file)

```python
"""
Dynamic Pricing for a Shared E-Bike Station with Tabular Q-Learning
------------------------------------------------------------------
States:
  (t_slot, inv_bin)
    - t_slot: 0..95 (96 x 15-minute steps per day)
    - inv_bin: discretized inventory level (0..INV_BINS-1)

Actions:
  price_tier in {0=Low, 1=Med, 2=High}

Reward:
  revenue_per_rental - penalties(stockout, leftover at day end)

Environment dynamics:
  rentals ~ demand_model(time, price) truncated by available inventory.
  inventory decreases by rentals each step. Day resets after 96 steps.

You’ll see:
- ε-greedy exploration vs exploitation
- Q-learning updates: Q(s,a) ← Q + α * (r + γ max_a' Q(s',a') - Q)
- Effect of γ, α, ε on learning
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# -------------------------
# Config & Hyperparameters
# -------------------------
SLOTS_PER_DAY = 96            # 96 * 15min = 24h
BIKE_CAPACITY = 40
INV_BINS = 41                 # 0..40 (no binning error; stays intuitive)
PRICE_TIERS = np.array([0, 1, 2])  # Low/Med/High
PRICE_VALUES = {0: 1.0, 1: 1.5, 2: 2.0}  # revenue per rental (arbitrary units)

# Demand controls
BASE_DEMAND = 3.0             # baseline rentals per 15 minutes at medium price
PEAK_BONUS_AM = 2.5           # morning commute boost
PEAK_BONUS_PM = 2.0           # evening commute boost
WEEKEND_MULT = 0.8            # weekends slightly lower (not used in base run)
PRICE_SENSITIVITY = 0.55      # higher → demand drops faster with price
NOISE_SD = 0.8                # stochasticity in demand

# Penalties
STOCKOUT_PENALTY = 1.0        # per customer turned away
LEFTOVER_PENALTY = 0.05       # per unused bike at end of day

# Q-learning hyperparams
ALPHA = 0.25
GAMMA = 0.98
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9995        # per episode
EPISODES = 2500               # try 2500-5000 for smoother convergence
EVAL_EVERY = 100

rng = np.random.default_rng(42)


# -------------------------
# Environment
# -------------------------
@dataclass
class StepResult:
    next_state: tuple
    reward: float
    done: bool
    info: dict


class BikePricingEnv:
    """
    One-day episodic environment.
    State: (t_slot, inv_bin) with inv_bin ∈ [0, BIKE_CAPACITY].
    Actions: 0 (Low), 1 (Med), 2 (High).
    """

    def __init__(self):
        self.capacity = BIKE_CAPACITY
        self.t = 0
        self.inv = self.capacity  # start full each morning

    def reset(self):
        self.t = 0
        self.inv = self.capacity
        return (self.t, self.inv)

    def demand_mean(self, t_slot, price_tier):
        """
        Mean rentals demand (before truncation by inventory) as a function of time and price.
        - Time profile: morning peak (8–10am), evening peak (5–7pm)
        - Price effect: multiplicative decay with tier
        """
        hour = (t_slot * 15) / 60.0  # 0..24
        # base curve
        mean = BASE_DEMAND

        # morning peak (8–10)
        if 8 <= hour < 10:
            mean += PEAK_BONUS_AM
        # evening peak (17–19)
        if 17 <= hour < 19:
            mean += PEAK_BONUS_PM

        # price sensitivity (higher tier → lower demand)
        mean *= np.exp(-PRICE_SENSITIVITY * price_tier)
        return max(0.0, mean)

    def step(self, action):
        price_tier = int(action)
        mean = self.demand_mean(self.t, price_tier)

        # stochastic demand (non-negative)
        demand = max(0.0, rng.normal(loc=mean, scale=NOISE_SD))

        # fulfilled rentals can’t exceed inventory
        rentals = int(min(self.inv, np.floor(demand)))
        turned_away = max(0, int(np.floor(demand)) - rentals)

        revenue = rentals * PRICE_VALUES[price_tier]
        penalty = turned_away * STOCKOUT_PENALTY
        reward = revenue - penalty

        self.inv -= rentals
        self.t += 1

        done = self.t >= SLOTS_PER_DAY
        info = {
            "rentals": rentals,
            "turned_away": turned_away,
            "revenue": revenue,
        }

        # End-of-day leftover penalty shaping
        if done and self.inv > 0:
            leftover_pen = LEFTOVER_PENALTY * self.inv
            reward -= leftover_pen
            info["leftover_penalty"] = leftover_pen

        next_state = (min(self.t, SLOTS_PER_DAY - 1), self.inv) if not done else (SLOTS_PER_DAY - 1, self.inv)
        return StepResult(next_state, reward, done, info)


# -------------------------
# Q-Learning Agent
# -------------------------
class QLearningAgent:
    def __init__(self, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON_START):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((SLOTS_PER_DAY, INV_BINS, len(PRICE_TIERS)), dtype=np.float32)

    def select_action(self, state, greedy=False):
        t, inv = state
        if (not greedy) and (rng.random() < self.epsilon):
            return int(rng.integers(low=0, high=len(PRICE_TIERS)))
        q_vals = self.Q[t, inv, :]
        return int(np.argmax(q_vals))

    def update(self, s, a, r, s_next):
        t, inv = s
        t2, inv2 = s_next
        best_next = np.max(self.Q[t2, inv2, :])
        td_target = r + self.gamma * best_next
        td_error = td_target - self.Q[t, inv, a]
        self.Q[t, inv, a] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)


# -------------------------
# Training Loop
# -------------------------
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
    return np.mean([run_episode(env, agent, greedy=True) for _ in range(episodes)])


def main():
    env = BikePricingEnv()
    agent = QLearningAgent()

    rewards = []
    evals = []

    for ep in range(1, EPISODES + 1):
        ep_ret = run_episode(env, agent, greedy=False)
        agent.decay_epsilon()
        rewards.append(ep_ret)

        if ep % EVAL_EVERY == 0:
            avg_eval = evaluate(env, agent, episodes=10)
            evals.append((ep, avg_eval))
            print(f"Episode {ep:4d} | TrainReturn={np.mean(rewards[-EVAL_EVERY:]):.2f} "
                  f"| EvalReturn={avg_eval:.2f} | ε={agent.epsilon:.3f}")

    # ---- Plots ----
    plt.figure()
    window = 50
    moving = np.convolve(rewards, np.ones(window)/window, mode="valid")
    plt.plot(rewards, alpha=0.3, label="Episode return")
    plt.plot(range(window-1, len(rewards)), moving, label=f"Moving avg ({window})")
    if evals:
        xs, ys = zip(*evals)
        plt.plot(xs, ys, marker="o", label="Greedy eval")
    plt.title("Training progress")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.show()

    # Visualize learned value at a few time slices (greedy Q)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
    times_to_show = [8*4, 12*4, 17*4, 22*4]  # 8:00, 12:00, 17:00, 22:00
    for ax, t in zip(axes.ravel(), times_to_show):
        q_slice = agent.Q[t, :, :]
        best_q = np.max(q_slice, axis=1)
        ax.plot(best_q)
        ax.set_title(f"t={t} (hour≈{(t*15)/60:.0f}) best Q vs inventory")
        ax.set_xlabel("Inventory")
        ax.set_ylabel("Best Q")
    plt.tight_layout()
    plt.show()

    # Print a greedy policy snapshot for a morning-peak hour across inventory
    t = 8*4  # 8:00 AM
    greedy_actions = np.argmax(agent.Q[t, :, :], axis=1)
    action_names = {0:"Low", 1:"Med", 2:"High"}
    print("\nGreedy price policy at 08:00 by inventory:")
    for inv in range(0, BIKE_CAPACITY + 1, 5):
        print(f"  inv={inv:2d} -> {action_names[int(greedy_actions[inv])]}")


if __name__ == "__main__":
    main()
```

---

## Step-by-step guide (what to look for)

1. **Run training** and watch the console every \~100 episodes:

   * **TrainReturn** rising: the agent is learning a better pricing policy.
   * **EvalReturn** (greedy) should improve more smoothly.
   * **ε (epsilon)** decays toward 0.05 → more exploitation later.
2. **Plots**:

   * The training curve (per-episode return and moving average) should trend upward.
   * The **best-Q vs inventory** plots: during peak hours (e.g., 8:00, 17:00), you’ll often see higher optimal Q at **medium/high price** when inventory is low (conserve bikes), and **lower price** when inventory is high (sell more).
3. **Greedy policy at 08:00**:

   * Inspect how the optimal price changes as inventory varies: higher price at low inventory; lower price at high inventory.

---

## How this maps to RL theory (in practice)

* **State (s):** `(time slot, inventory)` captures the situation relevant to pricing.
* **Action (a):** chosen price tier.
* **Transition (P):** simulator stochastically draws demand → subtracts rentals from inventory.
* **Reward (r):** revenue minus penalties (stockouts & leftover bikes). This is **reward shaping**—guides learning toward business goals.
* **Policy (π):** the ε-greedy rule during training; the **greedy** rule at evaluation.
* **Q-Learning (off-policy TD):** updates toward `r + γ max_a' Q(s',a')`.
* **Exploration/exploitation:** ε starts at 1.0 (random), decays to 0.05 (mostly greedy).
* **Discount (γ):** with `γ=0.98`, the agent values future inventory health (don’t sell too cheaply early and run out before evening peaks).

---

## Tuning & experiments (highly recommended)

* **Faster/slower ε decay:** change `EPSILON_DECAY` (e.g., 0.999 → explores more, slower convergence).
* **More stochastic demand:** raise `NOISE_SD` and see if learning becomes noisier.
* **Harsher stockout penalty:** increase `STOCKOUT_PENALTY` to force conservative pricing at low inventory.
* **Leftover penalty 0:** set `LEFTOVER_PENALTY=0` to see the agent push sales aggressively late in the day.
* **γ sensitivity:** try `GAMMA=0.9` vs `0.99` to change foresight.

---

## Why this single exercise is “real-world recent”

Micromobility and shared fleets (bikes/scooters) use **dynamic pricing** to balance **utilization** vs **customer satisfaction** under **stochastic demand**—a common, current industry problem. Your simulator captures the essence, and the same framework extends to:

* surge pricing for ridesharing,
* dynamic discounts in e-commerce,
* yield management for parking/EV charging.

---

## What you need (tools recap)

* **Python 3.9+** (macOS built-in or via `brew`)
* **numpy, matplotlib** (installed above)
* A single script file: `bike_pricing_qlearning.py`

---

## Next steps (optional when you’re ready)

* Swap tabular Q for a small **DQN** (neural net) when you add more state (e.g., weather, weekday/weekend, recent demand).
* Log detailed episode metrics to CSV and plot **learning curves** per time-of-day.
* Introduce **inventory restock** at noon to create a more complex MDP.
* Add **context (contextual RL)** like rain probability to condition pricing.

---

If you want, I can extend this with a tiny DQN version (still beginner-friendly), or add CSV logging + Seaborn-free plots for deeper analysis.
