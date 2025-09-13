Perfect — here are three small, macOS-friendly Python files that give you:

1. **CSV logging for DQN** (mirrors your Q-learning logger)
2. **Checkpoint save/load** (resume training; keep both `last.pt` and `best.pt`)
3. **A hyperparameter sweep** (compare ε-decay and penalties, with a results CSV)

They only need: `numpy`, `matplotlib`, `torch` (CPU). No seaborn.

---

# 0) Install (once)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy matplotlib
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

# 1) `env_bike_pricing.py` (shared environment)

Save this as **env\_bike\_pricing.py**.

```python
# env_bike_pricing.py
import math
import numpy as np
from dataclasses import dataclass

# ----- Core environment knobs -----
SLOTS_PER_DAY = 96           # 96 * 15min = 24h
BIKE_CAPACITY = 40
PRICE_VALUES = {0: 1.0, 1: 1.5, 2: 2.0}

# Demand profile (you can tweak these from the trainer via setters)
BASE_DEMAND = 3.0
PEAK_BONUS_AM = 2.5          # 8–10 am
PEAK_BONUS_PM = 2.0          # 5–7 pm
PRICE_SENSITIVITY = 0.55
NOISE_SD = 0.8

@dataclass
class StepResult:
    next_state: np.ndarray
    reward: float
    done: bool
    info: dict

class BikePricingEnv:
    """
    State: [one-hot time(96), inventory_fraction] => shape (97,)
    Actions: 0 (Low), 1 (Med), 2 (High)
    Reward: revenue - stockout_penalty*turned_away - end_of_day_leftover_penalty*leftover
    """
    def __init__(self, stockout_penalty=1.0, leftover_penalty=0.05, seed=1234):
        self.capacity = BIKE_CAPACITY
        self.stockout_penalty = float(stockout_penalty)
        self.leftover_penalty = float(leftover_penalty)
        self.rng = np.random.default_rng(int(seed))
        self.t = 0
        self.inv = self.capacity

    def set_seed(self, seed):
        self.rng = np.random.default_rng(int(seed))

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
        time_oh = np.zeros(SLOTS_PER_DAY, dtype=np.float32)
        time_oh[self.t] = 1.0
        inv_norm = np.array([self.inv / float(self.capacity)], dtype=np.float32)
        return np.concatenate([time_oh, inv_norm], axis=0)  # (97,)

    def step(self, action):
        price_tier = int(action)
        mean = self.demand_mean(self.t, price_tier)
        demand = max(0.0, self.rng.normal(loc=mean, scale=NOISE_SD))

        rentals = int(min(self.inv, math.floor(demand)))
        turned_away = max(0, int(math.floor(demand)) - rentals)

        revenue = rentals * PRICE_VALUES[price_tier]
        reward = revenue - self.stockout_penalty * turned_away

        self.inv -= rentals
        self.t += 1
        done = self.t >= SLOTS_PER_DAY

        if done and self.inv > 0:
            reward -= self.leftover_penalty * self.inv

        return StepResult(
            next_state=self._obs(),
            reward=reward,
            done=done,
            info={
                "rentals": rentals,
                "turned_away": turned_away,
                "revenue": revenue
            }
        )

def state_dim():
    return SLOTS_PER_DAY + 1  # 96 one-hot + 1 inventory fraction

def n_actions():
    return 3
```

---

# 2) `dqn_train.py` (CSV logger + checkpoints + resume)

Save this as **dqn\_train.py**.

```python
# dqn_train.py
import os, csv, argparse, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
from dataclasses import dataclass

from env_bike_pricing import BikePricingEnv, state_dim, n_actions

# -------------------------
# Defaults (override via CLI)
# -------------------------
GAMMA_DEFAULT = 0.98
LR_DEFAULT = 1e-3
BATCH_SIZE_DEFAULT = 64
REPLAY_CAP_DEFAULT = 50_000
TRAIN_START_DEFAULT = 1_000
TARGET_UPDATE_EVERY_DEFAULT = 1_000
EPISODES_DEFAULT = 2000

EPSILON_START_DEFAULT = 1.0
EPSILON_MIN_DEFAULT = 0.05
EPSILON_DECAY_DEFAULT = 0.9995

GRAD_CLIP_NORM_DEFAULT = 5.0
EVAL_EVERY_DEFAULT = 100
EVAL_EPISODES_DEFAULT = 10

DEVICE = torch.device("cpu")

@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    d: float

class ReplayBuffer:
    def __init__(self, capacity:int, seed:int=123):
        self.buf = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def push(self, s,a,r,s2,d):
        self.buf.append(Transition(s,a,r,s2,d))

    def sample(self, batch_size:int):
        idx = self.rng.choice(len(self.buf), size=batch_size, replace=False)
        batch = [self.buf[i] for i in idx]
        s = np.stack([t.s for t in batch])
        a = np.array([t.a for t in batch])
        r = np.array([t.r for t in batch], dtype=np.float32)
        s2 = np.stack([t.s2 for t in batch])
        d = np.array([t.d for t in batch], dtype=np.float32)
        return s,a,r,s2,d

    def __len__(self): return len(self.buf)

class QNet(nn.Module):
    def __init__(self, state_dim:int, n_actions:int, hidden:int=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
    def forward(self, x): return self.net(x)

class DQNAgent:
    def __init__(self, s_dim:int, a_dim:int, lr:float, gamma:float,
                 eps_start:float, eps_min:float, eps_decay:float, grad_clip:float):
        self.gamma = gamma
        self.epsilon = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.grad_clip = grad_clip

        self.q = QNet(s_dim, a_dim).to(DEVICE)
        self.target = QNet(s_dim, a_dim).to(DEVICE)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    def select_action(self, s:np.ndarray, greedy:bool=False):
        if (not greedy) and (np.random.rand() < self.epsilon):
            return int(np.random.randint(0, 3))
        with torch.no_grad():
            t = torch.from_numpy(s).float().to(DEVICE).unsqueeze(0)
            qv = self.q(t)
            return int(torch.argmax(qv, dim=1).item())

    def train_step(self, batch):
        s,a,r,s2,d = batch
        s   = torch.from_numpy(s).float().to(DEVICE)
        a   = torch.from_numpy(a).long().to(DEVICE).unsqueeze(1)
        r   = torch.from_numpy(r).float().to(DEVICE).unsqueeze(1)
        s2  = torch.from_numpy(s2).float().to(DEVICE)
        d   = torch.from_numpy(d).float().to(DEVICE).unsqueeze(1)

        q_sa = self.q(s).gather(1, a)
        with torch.no_grad():
            max_q_s2 = self.target(s2).max(dim=1, keepdim=True).values
            target = r + (1.0 - d) * self.gamma * max_q_s2
        loss = self.loss_fn(q_sa, target)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
        self.opt.step()
        return float(loss.item())

    def update_target(self):
        self.target.load_state_dict(self.q.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

def evaluate(env:BikePricingEnv, agent:DQNAgent, episodes:int):
    rets = []
    for _ in range(episodes):
        s = env.reset()
        ret = 0.0
        done = False
        while not done:
            a = agent.select_action(s, greedy=True)
            step = env.step(a)
            s = step.next_state
            ret += step.reward
            done = step.done
        rets.append(ret)
    return float(np.mean(rets))

def save_checkpoint(path:str, agent:DQNAgent, episode:int, total_steps:int, rng_state):
    ckpt = {
        "q": agent.q.state_dict(),
        "target": agent.target.state_dict(),
        "opt": agent.opt.state_dict(),
        "epsilon": agent.epsilon,
        "episode": episode,
        "total_steps": total_steps,
        "rng_state": rng_state,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)

def load_checkpoint(path:str, agent:DQNAgent):
    ckpt = torch.load(path, map_location=DEVICE)
    agent.q.load_state_dict(ckpt["q"])
    agent.target.load_state_dict(ckpt["target"])
    agent.opt.load_state_dict(ckpt["opt"])
    agent.epsilon = float(ckpt["epsilon"])
    return ckpt["episode"], ckpt["total_steps"], ckpt.get("rng_state", None)

def train_one_run(cfg):
    # ----- env & agent -----
    env = BikePricingEnv(stockout_penalty=cfg.stockout_penalty,
                         leftover_penalty=cfg.leftover_penalty,
                         seed=cfg.seed)
    s_dim = state_dim()
    a_dim = n_actions()

    agent = DQNAgent(
        s_dim, a_dim,
        lr=cfg.lr, gamma=cfg.gamma,
        eps_start=cfg.eps_start, eps_min=cfg.eps_min,
        eps_decay=cfg.eps_decay, grad_clip=cfg.grad_clip
    )
    replay = ReplayBuffer(cfg.replay_cap, seed=cfg.seed)

    # ----- logging paths -----
    os.makedirs(cfg.log_dir, exist_ok=True)
    log_csv = os.path.join(cfg.log_dir, "log.csv")
    last_ckpt = os.path.join(cfg.log_dir, "last.pt")
    best_ckpt = os.path.join(cfg.log_dir, "best.pt")

    # ----- resume if requested -----
    start_ep = 1
    total_steps = 0
    best_eval = -1e18
    if cfg.resume and os.path.isfile(cfg.resume):
        start_ep, total_steps, rng_state = load_checkpoint(cfg.resume, agent)
        if rng_state is not None:
            np.random.set_state(rng_state["numpy"])
            torch.random.set_rng_state(rng_state["torch"])
        print(f"[Resume] Loaded from {cfg.resume} at episode={start_ep}, steps={total_steps}")
        start_ep += 1

    # ----- CSV header -----
    if not (cfg.resume and os.path.isfile(log_csv)):
        with open(log_csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["episode","train_return","eval_return","epsilon","loss"])

    # ----- training -----
    train_returns, eval_x, eval_y, losses = [], [], [], []
    for ep in range(start_ep, cfg.episodes + 1):
        s = env.reset()
        ep_ret, ep_loss = 0.0, []
        done = False

        while not done:
            a = agent.select_action(s, greedy=False)
            step = env.step(a)
            replay.push(s, a, step.reward, step.next_state, float(step.done))
            s = step.next_state
            ep_ret += step.reward
            done = step.done

            total_steps += 1
            if len(replay) >= cfg.train_start:
                batch = replay.sample(cfg.batch_size)
                loss = agent.train_step(batch)
                ep_loss.append(loss)
                if total_steps % cfg.target_update_every == 0:
                    agent.update_target()

        agent.decay_epsilon()
        avg_loss = float(np.mean(ep_loss)) if ep_loss else 0.0
        train_returns.append(ep_ret)
        losses.append(avg_loss)

        eval_ret = ""
        if ep % cfg.eval_every == 0:
            eval_ret = evaluate(env, agent, cfg.eval_episodes)
            eval_x.append(ep); eval_y.append(eval_ret)
            if eval_ret > best_eval:
                best_eval = eval_ret
                save_checkpoint(best_ckpt, agent, ep, total_steps, {
                    "numpy": np.random.get_state(),
                    "torch": torch.random.get_rng_state(),
                })
            print(f"Ep {ep:4d} | TrainAvg({cfg.eval_every})={np.mean(train_returns[-cfg.eval_every:]):.2f} "
                  f"| Eval={eval_ret:.2f} | Loss={np.mean(losses[-cfg.eval_every:]):.4f} | ε={agent.epsilon:.3f}")

        # append to CSV every episode
        with open(log_csv, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([ep, f"{ep_ret:.4f}", f"{eval_ret if eval_ret!='' else ''}", f"{agent.epsilon:.6f}", f"{avg_loss:.6f}"])

        # save "last" checkpoint occasionally (or every episode)
        save_checkpoint(last_ckpt, agent, ep, total_steps, {
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
        })

    # ----- plots -----
    plt.figure()
    plt.plot(train_returns, alpha=0.35, label="Episode return")
    if len(train_returns) >= 50:
        win = 50
        mov = np.convolve(train_returns, np.ones(win)/win, mode="valid")
        plt.plot(range(win-1, len(train_returns)), mov, label=f"Moving avg ({win})")
    if eval_x:
        plt.plot(eval_x, eval_y, marker="o", label="Evaluation (greedy)")
    plt.title("DQN: Training Progress")
    plt.xlabel("Episode"); plt.ylabel("Return"); plt.legend(); plt.tight_layout()
    plt.show()

    return {
        "best_eval": best_eval if best_eval != -1e18 else None,
        "last_train_avg": float(np.mean(train_returns[-min(100, len(train_returns)):])),
        "log_csv": log_csv,
        "best_ckpt": best_ckpt,
        "last_ckpt": last_ckpt,
    }

def parse_args():
    p = argparse.ArgumentParser(description="Bike Pricing DQN with CSV & Checkpoints")
    p.add_argument("--log-dir", type=str, default="runs/dqn_default", help="Where to store CSV & checkpoints")
    p.add_argument("--episodes", type=int, default=EPISODES_DEFAULT)
    p.add_argument("--gamma", type=float, default=GAMMA_DEFAULT)
    p.add_argument("--lr", type=float, default=LR_DEFAULT)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    p.add_argument("--replay-cap", type=int, default=REPLAY_CAP_DEFAULT)
    p.add_argument("--train-start", type=int, default=TRAIN_START_DEFAULT)
    p.add_argument("--target-update-every", type=int, default=TARGET_UPDATE_EVERY_DEFAULT)
    p.add_argument("--eps-start", type=float, default=EPSILON_START_DEFAULT)
    p.add_argument("--eps-min", type=float, default=EPSILON_MIN_DEFAULT)
    p.add_argument("--eps-decay", type=float, default=EPSILON_DECAY_DEFAULT)
    p.add_argument("--grad-clip", type=float, default=GRAD_CLIP_NORM_DEFAULT)
    p.add_argument("--eval-every", type=int, default=EVAL_EVERY_DEFAULT)
    p.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES_DEFAULT)
    p.add_argument("--stockout-penalty", type=float, default=1.0)
    p.add_argument("--leftover-penalty", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume (e.g., runs/.../last.pt)")
    return p.parse_args()

if __name__ == "__main__":
    cfg = parse_args()
    res = train_one_run(cfg)
    print("\n==> Summary")
    for k,v in res.items(): print(f"{k}: {v}")
```

### Run examples

```bash
# fresh run (default config)
python dqn_train.py --log-dir runs/dqn_baseline

# resume from last checkpoint
python dqn_train.py --log-dir runs/dqn_baseline --resume runs/dqn_baseline/last.pt

# try a slower epsilon decay and harsher stockout penalty
python dqn_train.py --log-dir runs/eps0999_pen2 --eps-decay 0.999 --stockout-penalty 2.0
```

**What you get in `--log-dir`:**

* `log.csv` → per-episode `train_return`, periodic `eval_return`, `epsilon`, `loss`
* `last.pt`  → rolling checkpoint (for resume)
* `best.pt`  → best evaluation checkpoint

---

# 3) `sweep.py` (hyperparameter sweep with CSV summary)

This script runs multiple configurations **in-process** (no subprocess), each in its own `log_dir`. It records the **best evaluation return** per config in a sweep CSV and draws a simple Matplotlib chart (no seaborn).

Save as **sweep.py**.

```python
# sweep.py
import os, csv, itertools, shutil
from datetime import datetime

from dqn_train import train_one_run
from types import SimpleNamespace

def run_sweep(base_dir="runs/sweep", seeds=(1234,), eps_decays=(0.9995, 0.999, 0.998),
              stockout_penalties=(1.0, 2.0), leftover_penalties=(0.05,),
              episodes=1200):
    os.makedirs(base_dir, exist_ok=True)
    results = []
    i = 0

    for seed, eps_decay, spen, lpen in itertools.product(seeds, eps_decays, stockout_penalties, leftover_penalties):
        i += 1
        run_name = f"seed{seed}_eps{eps_decay}_sp{spen}_lp{lpen}"
        log_dir = os.path.join(base_dir, run_name)
        print(f"\n=== [{i}] {run_name} ===")

        # Build a config namespace compatible with train_one_run
        cfg = SimpleNamespace(
            log_dir=log_dir,
            episodes=episodes,
            gamma=0.98,
            lr=1e-3,
            batch_size=64,
            replay_cap=50_000,
            train_start=1_000,
            target_update_every=1_000,
            eps_start=1.0,
            eps_min=0.05,
            eps_decay=float(eps_decay),
            grad_clip=5.0,
            eval_every=100,
            eval_episodes=10,
            stockout_penalty=float(spen),
            leftover_penalty=float(lpen),
            seed=int(seed),
            resume="",
        )

        res = train_one_run(cfg)
        results.append({
            "run": run_name,
            "best_eval": res["best_eval"],
            "last_train_avg": res["last_train_avg"],
            "log_csv": res["log_csv"],
            "best_ckpt": res["best_ckpt"],
        })

    # Write sweep summary CSV
    summary_csv = os.path.join(base_dir, "sweep_results.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run","best_eval","last_train_avg","log_csv","best_ckpt"])
        for r in results:
            w.writerow([r["run"], r["best_eval"], r["last_train_avg"], r["log_csv"], r["best_ckpt"]])

    print(f"\nSweep complete. Summary: {summary_csv}")

    # Optional: quick bar chart of best_eval
    try:
        import matplotlib.pyplot as plt
        labels = [r["run"] for r in results]
        vals = [r["best_eval"] if r["best_eval"] is not None else float("nan") for r in results]
        plt.figure(figsize=(max(6, len(labels)*0.6), 4))
        plt.bar(range(len(labels)), vals)
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.ylabel("Best Eval Return")
        plt.title("DQN Sweep: Best Evaluation Return per Run")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Plotting skipped: {e}")

if __name__ == "__main__":
    # Example sweep: 3 epsilon decays × 2 stockout penalties = 6 runs
    run_sweep(
        base_dir="runs/sweep",
        seeds=(1234,),
        eps_decays=(0.9995, 0.999, 0.998),
        stockout_penalties=(1.0, 2.0),
        leftover_penalties=(0.05,),
        episodes=1200
    )
```

### Run the sweep

```bash
python sweep.py
# -> runs/sweep/sweep_results.csv + per-run folders with individual logs & checkpoints
```

---

## How to use these pieces (quick workflow)

1. **Single run first**

   ```bash
   python dqn_train.py --log-dir runs/dqn_baseline
   ```

   Inspect `runs/dqn_baseline/log.csv`, `best.pt`, training plot.

2. **Resume later**

   ```bash
   python dqn_train.py --log-dir runs/dqn_baseline --resume runs/dqn_baseline/last.pt
   ```

3. **Compare strategies**

   ```bash
   python dqn_train.py --log-dir runs/eps0999_pen2 --eps-decay 0.999 --stockout-penalty 2.0
   ```

4. **Batch compare (sweep)**

   ```bash
   python sweep.py
   ```

   Open `runs/sweep/sweep_results.csv`. The bar chart helps you spot the best combo.

---

## What you’ll learn with these additions

* **Experiment hygiene**: every run is logged to CSV; you can compute moving averages or compare runs.
* **Reproducibility**: checkpoints let you pause/resume and share models.
* **Hyperparameter intuition**:

  * **ε-decay**: slower decay (e.g., 0.999) explores longer → sometimes higher final returns.
  * **Stockout penalty**: larger → conservative pricing at low inventory; watch how policies shift.
  * **Leftover penalty**: smaller → aggressive late-day selling.

If you want, I can also add a tiny **`evaluate.py`** that loads `best.pt` and prints a deterministic evaluation report, or a **plotter** that overlays multiple `log.csv` files on one chart.
