# Snake Reinforcement Learning System

A complete reinforcement learning pipeline for training agents to play the game Snake using **DQN** and **PPO** algorithms with three custom innovations.

**Tech Stack:** Python 3, PyTorch, NumPy, Pygame, Matplotlib

---

## Overview

This project implements a modular RL system for training Snake-playing agents with the following key features:

- **Gym-like Snake Environment** with configurable grid size and step limits
- **DQN Agent** with experience replay and target network
- **PPO Agent** with actor-critic architecture and GAE
- **Three Required Innovations:**
  1. **Distance-based shaping reward** - Encourages moving toward food
  2. **Anti-loop / no-progress penalty** - Discourages repetitive behavior
  3. **Adaptive PPO clipping schedule** - Dynamic clipping parameter decay
- **Complete training and evaluation pipeline**
- **Visualization tools** for learning curves and gameplay

---

## Installation

### Prerequisites

- Python 3.7+
- pip

### Install Dependencies

```bash
pip install torch numpy matplotlib pandas
```

**Optional (for visualization):**

```bash
pip install pygame
```

---

## Project Structure

```
snake_rl/
├── env/
│   ├── snake_env.py          # Snake environment with Gym-like API
│   └── __init__.py
├── agents/
│   ├── dqn/
│   │   ├── dqn_agent.py      # DQN agent implementation
│   │   ├── q_network.py      # Q-network architecture
│   │   ├── replay_buffer.py  # Experience replay buffer
│   │   └── __init__.py
│   ├── ppo/
│   │   ├── ppo_agent.py      # PPO agent implementation
│   │   ├── actor_critic.py   # Actor-Critic network
│   │   └── __init__.py
│   └── __init__.py
├── training/
│   ├── train_dqn.py          # DQN training script
│   ├── train_ppo.py          # PPO training script
│   └── __init__.py
├── evaluation/
│   ├── evaluate_agent.py     # Agent evaluation script
│   ├── plot_learning_curves.py  # Visualization script
│   └── __init__.py
├── utils/
│   ├── config.py             # Configuration management
│   ├── logger.py             # Metrics logging
│   ├── scheduler.py          # Parameter schedulers
│   └── __init__.py
├── models/                   # Saved model checkpoints
└── plots/                    # Generated plots
```

---

## Quick Start

### 1. Train DQN Agent

Train the DQN agent for 2000 episodes (default):

```bash
python snake_rl/training/train_dqn.py
```

Or specify a custom number of episodes:

```bash
python snake_rl/training/train_dqn.py --episodes 5000
```

**Output:**
- Model checkpoints saved to `snake_rl/models/`
- Training log saved to `snake_rl/models/dqn_training_log.csv`

### 2. Train PPO Agent

Train the PPO agent for 2000 episodes (default):

```bash
python snake_rl/training/train_ppo.py
```

Or specify a custom number of episodes:

```bash
python snake_rl/training/train_ppo.py --episodes 5000
```

**Output:**
- Model checkpoints saved to `snake_rl/models/`
- Training log saved to `snake_rl/models/ppo_training_log.csv`

### 3. Evaluate Agent

Evaluate a trained agent over 100 episodes:

```bash
python snake_rl/evaluation/evaluate_agent.py --agent ppo
```

Or evaluate DQN:

```bash
python snake_rl/evaluation/evaluate_agent.py --agent dqn
```

**With Pygame Visualization:**

```bash
python snake_rl/evaluation/evaluate_agent.py --agent ppo --render
```

**Custom Options:**

```bash
python snake_rl/evaluation/evaluate_agent.py --agent ppo --model snake_rl/models/ppo_episode_1000.pt --episodes 50 --render
```

### 4. Plot Learning Curves

Generate visualization plots:

```bash
python snake_rl/evaluation/plot_learning_curves.py
```

**Options:**

```bash
# Plot only DQN
python snake_rl/evaluation/plot_learning_curves.py --agent dqn

# Plot only PPO
python snake_rl/evaluation/plot_learning_curves.py --agent ppo

# Plot both (default)
python snake_rl/evaluation/plot_learning_curves.py --agent both
```

**Output:**
- Plots saved to `snake_rl/plots/` directory

---

## Custom Innovations

### Innovation 1: Distance-Based Shaping Reward

**Purpose:** Guide the snake toward food by providing incremental feedback.

**Implementation:**
- Calculate Manhattan distance to food before and after each step
- Reward: `+0.1` if distance decreases (getting closer)
- Penalty: `-0.1` if distance increases (getting farther)

**Location:** `snake_rl/env/snake_env.py` in the `step()` method

**Code:**
```python
current_distance = self._get_distance_to_food()
if current_distance < self.prev_distance:
    reward += 0.1  # Getting closer
elif current_distance > self.prev_distance:
    reward -= 0.1  # Getting farther
```

### Innovation 2: Anti-Loop / No-Progress Penalty

**Purpose:** Discourage the snake from looping in the same area without making progress.

**Implementation:**
- Track the last 8 positions visited
- If current position is in recent history: penalty of `-1.0`
- If no food eaten for 100+ steps: penalty of `-1.0` and terminate episode

**Location:** `snake_rl/env/snake_env.py` in the `step()` method

**Code:**
```python
if new_head in self.position_history:
    reward -= 1.0  # Loop penalty

if self.steps_without_food > 100:
    reward -= 1.0
    done = True
```

### Innovation 3: Adaptive PPO Clipping Schedule

**Purpose:** Start with aggressive policy updates and gradually become more conservative.

**Implementation:**
- Exponential decay schedule: `clip = max(0.1, 0.3 * exp(-0.0001 * update_count))`
- Starts at `0.3` and decays to minimum of `0.1`
- Applied to PPO's ratio clipping

**Location:** `snake_rl/utils/scheduler.py` and used in `snake_rl/training/train_ppo.py`

**Code:**
```python
clip_epsilon = clip_scheduler.get_clip_epsilon(agent.update_count)
```

---

## Configuration

All hyperparameters are defined in `snake_rl/utils/config.py`:

### Environment Configuration
- `grid_size`: 10 (10x10 grid)
- `max_steps`: 200 (maximum steps per episode)

### DQN Configuration
- `learning_rate`: 0.001
- `gamma`: 0.99
- `epsilon_start`: 1.0
- `epsilon_end`: 0.01
- `epsilon_decay_steps`: 10000
- `replay_buffer_size`: 10000
- `batch_size`: 64
- `target_update_frequency`: 100

### PPO Configuration
- `learning_rate`: 0.0003
- `gamma`: 0.99
- `gae_lambda`: 0.95
- `clip_epsilon_start`: 0.3
- `clip_epsilon_end`: 0.1
- `n_epochs`: 10
- `batch_size`: 64
- `rollout_steps`: 2048

**To modify configurations**, edit the values in `snake_rl/utils/config.py`.

---

## State Representation

The environment provides a **9-dimensional state vector**:

1. **Relative food X** (normalized): Horizontal distance to food
2. **Relative food Y** (normalized): Vertical distance to food
3. **Danger front**: Binary flag for danger straight ahead
4. **Danger left**: Binary flag for danger to the left
5. **Danger right**: Binary flag for danger to the right
6-9. **Direction one-hot**: Current direction (up/right/down/left)

## Action Space

The agent can take **3 discrete actions**:
- `0`: Turn left
- `1`: Go straight
- `2`: Turn right

## Reward Structure

- **+10**: Eating food
- **-10**: Collision (wall or self)
- **+0.1**: Moving closer to food (Innovation 1)
- **-0.1**: Moving farther from food (Innovation 1)
- **-1.0**: Loop detected or no progress (Innovation 2)

---

## Training Details

### DQN Training
- Uses epsilon-greedy exploration (1.0 → 0.01 over 10000 steps)
- Experience replay with buffer size 10000
- Target network updated every 100 training steps
- Saves checkpoints every 500 episodes

### PPO Training
- Collects 2048 steps per rollout
- Uses Generalized Advantage Estimation (GAE)
- Updates policy for 10 epochs per rollout
- Adaptive clipping schedule (Innovation 3)
- Saves checkpoints every 500 episodes

---

## Expected Results

With default settings (2000 episodes):

**DQN:**
- Training time: ~15-30 minutes (CPU)
- Expected average score: 3-8 food items
- Exploration helps discover diverse strategies

**PPO:**
- Training time: ~20-40 minutes (CPU)
- Expected average score: 4-10 food items
- More stable learning with better long-term performance

**Note:** Results may vary based on hardware and random initialization.

---

## Troubleshooting

### ImportError: No module named 'snake_rl'

Make sure you're running scripts from the project root directory:

```bash
cd "C:\Users\User\Desktop\CS 175 Final Project"
python snake_rl/training/train_dqn.py
```

### PyGame not available

PyGame is optional and only needed for visualization:

```bash
pip install pygame
```

### CUDA out of memory

The code automatically uses CPU if CUDA is not available. To force CPU:

```python
# Edit agent initialization to use CPU
device = torch.device("cpu")
```

### Training too slow

Reduce the number of episodes or adjust:
- `max_steps` in environment config
- `rollout_steps` in PPO config (for PPO only)

---

## File Descriptions

### Core Components

- **`snake_env.py`**: Complete Snake game environment with Gym API
- **`dqn_agent.py`**: DQN agent with epsilon-greedy and experience replay
- **`ppo_agent.py`**: PPO agent with GAE and clipped objective
- **`q_network.py`**: Feed-forward neural network for Q-values
- **`actor_critic.py`**: Shared network for policy and value functions

### Training & Evaluation

- **`train_dqn.py`**: End-to-end DQN training pipeline
- **`train_ppo.py`**: End-to-end PPO training pipeline
- **`evaluate_agent.py`**: Evaluate trained agents with statistics
- **`plot_learning_curves.py`**: Generate publication-quality plots

### Utilities

- **`config.py`**: Centralized configuration management
- **`logger.py`**: Metrics tracking and CSV export
- **`scheduler.py`**: Epsilon decay and adaptive clipping

---

## Extending the System

### Adding New Rewards

Edit `snake_env.py` in the `step()` method:

```python
# Add custom reward logic
if custom_condition:
    reward += custom_value
```

### Modifying Network Architecture

Edit `q_network.py` or `actor_critic.py`:

```python
# Example: Add more layers
self.fc4 = nn.Linear(hidden_dim, hidden_dim)
```

### Creating Custom Agents

Implement the same interface:
- `select_action(state)`
- `train()` (for training)
- `save(filepath)`
- `load(filepath)`

---

## Acceptance Criteria

✅ Fully functional Snake environment  
✅ DQN and PPO agents implemented  
✅ All three innovations integrated:
  - Distance-based shaping reward
  - Loop penalty
  - Adaptive PPO clipping  
✅ End-to-end trainable  
✅ Model save/load functionality  
✅ Evaluation with statistics  
✅ Learning curve visualization  
✅ Modular, documented code  

---

## License

This project is created for educational purposes as part of CS 175 Final Project.


---

## Acknowledgments

- PyTorch for deep learning framework
- OpenAI Gym for environment design inspiration
- Snake game mechanics


