# Dueling Double DQN — Flappy Bird

A Deep Reinforcement Learning agent that learns to play Flappy Bird from scratch using **Dueling DQN** and **Double DQN**, built with PyTorch and the `flappy_bird_gymnasium` environment.

## Overview

This project trains an agent to play Flappy Bird purely through trial and error, using no hand-coded rules. It combines two well-known improvements over vanilla DQN:

- **Dueling DQN** — splits the Q-value estimation into separate value and advantage streams, leading to more stable learning.
- **Double DQN** — decouples action selection from action evaluation to reduce Q-value overestimation bias.

The agent uses lidar-based observations from the environment, an experience replay buffer for stable training, and epsilon-greedy exploration with decay.

## Results

The agent's best single-episode reward improved from **-7.5 to 144.6** over the first ~400K training episodes (training is ongoing toward 1M episodes to find its performance ceiling).

| Episode | Best Reward |
|---|---|
| 0 | -7.5 |
| 13,854 | 5.0 |
| 57,369 | 17.9 |
| 125,278 | 76.2 |
| 409,363 | 144.6 |

## Features

- Dueling DQN architecture (separate value/advantage streams)
- Double DQN target computation
- Experience replay buffer
- Epsilon-greedy exploration with exponential decay
- Target network synchronization
- YAML-based configurable hyperparameters (supports multiple environments/configs)
- Automated training logs, reward graphs, and best-model checkpointing

## Project Structure

```
.
├── new_agent.py            # Main training/evaluation loop
├── new_dqn.py               # Dueling DQN network architecture
├── experience_replay.py    # Replay memory buffer
├── hyperparameters.yml     # Configurable hyperparameter sets
└── runs/                    # Saved models, logs, and reward graphs
```

## Setup

```bash
pip install torch gymnasium flappy_bird_gymnasium pyyaml matplotlib numpy
```

## Usage

**Train the agent:**
```bash
python new_agent.py flappybird1 --train
```

**Run a trained model:**
```bash
python new_agent.py flappybird1
```

`flappybird1` refers to the hyperparameter set defined in `hyperparameters.yml`. You can add new sets (e.g. for CartPole or other Gymnasium environments) without touching the code.

## Hyperparameters

Key settings used for the Flappy Bird run (`hyperparameters.yml`):

| Parameter | Value |
|---|---|
| Learning rate | 0.0001 |
| Discount factor (γ) | 0.99 |
| Replay memory size | 100,000 |
| Mini-batch size | 32 |
| Epsilon decay | 0.99995 |
| Epsilon min | 0.05 |
| Network sync rate | 10 steps |
| Hidden layer size | 512 |
| Double DQN | Enabled |
| Dueling DQN | Enabled |

## Roadmap

- [x] Vanilla DQN baseline
- [x] Double DQN
- [x] Dueling DQN
- [ ] Prioritized Experience Replay
- [ ] Rainbow DQN

## Acknowledgements

Built using [`flappy_bird_gymnasium`](https://github.com/markub3327/flappy-bird-gymnasium) and PyTorch.
