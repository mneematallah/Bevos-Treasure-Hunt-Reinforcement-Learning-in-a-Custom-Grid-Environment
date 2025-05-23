# Bevo's Treasure Hunt 🐮💎  
_Reinforcement Learning in a Custom Grid World Environment_

**Author:** Moustafa Neematallah  
📖 [Blog Post](https://medium.com/@mneematallah2003/bevos-treasure-hunt-reinforcement-learning-in-a-custom-grid-world-background-582cb36580f4)  
🎥 [Demo Video](https://youtu.be/5pfCDMJudMg)

---

## 🧠 Overview

Bevo’s Treasure Hunt is a custom **Reinforcement Learning (RL)** project where an agent (Bevo 🐮) learns to navigate a 2D grid environment using the **PPO (Proximal Policy Optimization)** algorithm. The environment includes **positive rewards (grass)** and **negative obstacles (OU logos 🟥)**. The goal: teach Bevo to collect treasures efficiently while avoiding penalties and repetitive motion.

This project demonstrates:
- Custom OpenAI Gymnasium environment design
- Reward shaping and state-space engineering
- Deep RL training using Stable-Baselines3 PPO
- Video capture and performance visualization

---

## 🎮 Features

- ✅ Custom environment: `GridWorldEnv` using OpenAI Gym interface
- 🎯 Goal: collect grass (+100), avoid OU logos (-10), and minimize invalid moves
- 🔁 PPO-based policy trained using vectorized environments and observation normalization
- 📉 Termination logic includes score thresholds, max steps, and **oscillation detection**
- 🖼️ Visual rendering with **pygame** and **video export using OpenCV**
- 🧪 Supports both **training** and **evaluation** via command line

---

## 🗂️ Project Structure

| File | Description |
|------|-------------|
| `grid_world.py` | Core GridWorld environment definition (Gym API) |
| `train.py` | PPO training loop with custom neural network |
| `evaluation.py` | Visualize trained agent and save video episodes |
| `main.py` | CLI entry point to train, evaluate, or both |
| `test.py` | Sanity test for pygame setup |
| `images/` | Folder containing Bevo, OU, and grass sprites |

---

## ⚙️ Environment Details

- **State:** Flattened array with agent's (x, y) and grid tiles
- **Action Space:** `Discrete(4)` — up, down, left, right
- **Reward Design:**
  - `+100`: Grass
  - `-10`: OU logo
  - `-0.01`: Step cost
  - `-1`: Invalid move
- **Termination:**
  - Score below -20
  - All grass collected
  - Oscillation (looping movement)
  - Max step limit

---

Make sure to include images/ with:

bevo.png

ou.png

grass.png

## 🚀 Run the Project

Train and evaluate Bevo:
python main.py --mode both

Train only:
python main.py --mode train

Evaluate a trained model:
python main.py --mode evaluate

Make sure policy/ppo_gridworld_model.zip exists if evaluating only.

##📊 Results
Bevo successfully learns to:

Navigate the grid to collect all grass tiles

Avoid OU logo traps

Reduce invalid movement penalties

Escape oscillating movement patterns

The trained model was evaluated over multiple episodes and visualized in .mp4 videos.
✅ Agent performance improves steadily across episodes

## ✅ Installation

```bash
pip install pygame gymnasium stable-baselines3 opencv-python
