
# Dou Dizhu RL: Training and Playing Against Recurrent Learning Agents  

<!-- A LITTLE HEADING :D -->
Dou Dizhu RL is a reinforcement learning (RL) project that implements a simplified 1v1 version of Dou Dizhu (斗地主), a popular shedding card game.  
Learn more about it on [Wikipedia](https://en.wikipedia.org/wiki/Dou_dizhu)

> Shuffled and dealt many rounds of 斗地主 over dinner tables and sidewalks this (2025) sticky summer studying abroad in LDN, where I also took my first ML class! This bot pays homage to those youthful, graceless nights with my friends.

Inside this project!  
- A **gym-like RL environment** (`DoudizhuEnv`) for agent training.  
- Multiple **baseline bots** (random, greedy, conservative).  
- A **multi-stage PPO training pipeline** using **Stable Baselines3**.  
- A **terminal-based game** where humans can play against bots or the trained RL agent.  


<!-- ABOUT THE PROJECT SECTION -->
## About the Project  
RL agent's birth from environment engineering to multi-agent RL training, evaluation, and deployment.  

Key features:  
- **Game Engine**: Implements simplified Dou Dizhu rules (legal moves, role assignment, win detection).  
- **Baseline Agents**: Random, greedy, conservative, and human-input agents for benchmarking.  
- **RL Pipeline**: Multi-stage curriculum training (random opponents → mixed → greedy → self-play).  
- **Reinforcement Learning**: PPO agents trained with Stable Baselines3 (`stage1_random.py` → `stage4_selfplay.py`).  
- **Play CLI**: Interactive terminal game (`play_human_vs_bot.py`) where users can challenge rule-based bots or the trained RL model.  

### Built With  
- **Python 3.9+**  
- **Stable Baselines3 (2.7.0)** – PPO implementation for RL training  
- **Gym-like custom environment** – `DoudizhuEnv`  
- **NumPy** – state/action vector processing  
- **TensorBoard** – training visualization  
- **CLI (Python standard library)** – interactive gameplay against bots/agents  


<!-- GETTING STARTED SECTION -->
## Getting Started  

### Prerequisites  
- Python 3.9+  
- venv recommended

Install dependencies:  
```bash
pip install -r requirements.txt
```

### Installation
```bash
git clone https://github.com/your-username/doudizhu-rl.git
cd doudizhu-rl
```

<!-- USAGE SECTION -->
## Usage
### A. Train RL Agents
Reproduce trained agent by training across multiple stages:
```bash
# Stage 1: Train vs random bot
python training/stage1_random.py  

# Stage 2: Train vs mixed opponents
python training/stage2_mixed.py  

# Stage 3: Train vs greedy bot
python training/stage3_greedy.py  

# Stage 4: Self-play fine-tuning
python training/stage4_selfplay.py  

# Stage 5: More Self-play and Less Mixed-opponents
python training/stage5_mixed_selfplay.py
```

Trained models are saved under ```stages/```

With about 100k episodes:

![Final RL Training with 100k episode](https://github.com/evhuh/doudizhu-rl/raw/main/assets/readme_assets/final_rl_stats.png)

TensorBoard can be launched to visualize progress:
```bash
tensorboard --logdir runs/
```

### B. Play Against Bots or RL Agent
Launch the CLI
```bash
python play_human_vs_bot.py
```

Features:
- Choose opponents: Random, greedy, conservative, or rl
- Select your role (landlord, farmer, random)
- Interactive gameplay with real-time feedback

Simple Gameplay:

<a href="http://www.youtube.com/watch?feature=player_embedded&v=Dh9T4j9lw9I
" target="_blank"><img src="http://img.youtube.com/vi/Dh9T4j9lw9I/0.jpg" 
alt="Gameplay against RL Agent" width="420" height="315" border="10" /></a>

<!-- REFLECTION SECTION
## Reflections -->
