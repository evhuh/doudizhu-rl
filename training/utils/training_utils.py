# training/utils/training_utils.py

# Shared Utils for PPO training (DoudiZhu RL agent)
# - PPO config
# - Logging w role-specific metrics
# - Eval functions
# - Checkpt management



import os
import json
import numpy as np
import torch
from typing import Dict, Any
from collections import deque
import time
from dataclasses import dataclass
# import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback



# PPO Hyperparams for Diff Training Stages
@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    batch_size: int = 2048
    n_steps: int = 2048
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.1
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    policy_kwargs: Dict = None
    
    def __post_init__(self):
        if self.policy_kwargs is None:
            self.policy_kwargs = {
                'net_arch': {
                    'pi': [256, 256, 128], # Pplicy network
                    'vf': [256, 128] # value network
                },
                'activation_fn': torch.nn.Tanh
            }

# Metrics Tracking for Training
class DoudizhuMetrics:
    def __init__(self):
        self.reset()
    
    def reset(self):
        # Basic metrics
        self.episodes = 0
        self.wins = 0
        self.losses = 0
        self.timeouts = 0
        self.episode_rewards = []
        self.episode_lengths = []
        
        self.cards_remaining_on_loss = []
        self.opponent_cards_remaining = []
        
        # Role-specific metrics
        self.role_performance = {
            'landlord': {'wins': 0, 'losses': 0, 'games': 0},
            'farmer': {'wins': 0, 'losses': 0, 'games': 0}
        }
        self.game_length_by_role = {
            'landlord': [],
            'farmer': []
        }
        
        # Tactical metrics
        self.bombing_frequency = 0
        self.invalid_actions = 0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0
        
        # Rolling averages
        self.win_rate_window = deque(maxlen=100)
        self.recent_win_rates = []
        
        # Timing
        self.start_time = time.time()
    
    # Update Metrics w Episode Res'
    def update(self, info: Dict[str, Any], reward: float, episode_length: int):
        self.episodes += 1
        self.episode_rewards.append(reward)
        self.episode_lengths.append(episode_length)
        
        # Game outcome
        won = info.get('rl_agent_won', False)
        role = info.get('rl_agent_role', 'unknown')
        
        if won:
            self.wins += 1
            self.win_rate_window.append(1)
            self.consecutive_losses = 0
            if role in self.role_performance:
                self.role_performance[role]['wins'] += 1
        else:
            self.losses += 1
            self.win_rate_window.append(0)
            self.consecutive_losses += 1
            self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
            
            if role in self.role_performance:
                self.role_performance[role]['losses'] += 1
            
            # Track cards remaining on loss
            cards_remaining = info.get('cards_remaining', 0)
            self.cards_remaining_on_loss.append(cards_remaining)
        
        # Role-specific tracking
        if role in self.role_performance:
            self.role_performance[role]['games'] += 1
            self.game_length_by_role[role].append(episode_length)
        
        # Opponent cards at game end
        opp_cards = info.get('opponent_cards_remaining', 0)
        self.opponent_cards_remaining.append(opp_cards)
        
        # Tactical metrics
        if info.get('invalid_action', False):
            self.invalid_actions += 1
        
        # Track win rate over time
        if len(self.win_rate_window) >= 10:
            current_wr = sum(self.win_rate_window) / len(self.win_rate_window)
            self.recent_win_rates.append(current_wr)
    
    # Get Metrics Summary
    def get_summary(self) -> Dict[str, Any]:
        total_games = self.wins + self.losses
        win_rate = self.wins / max(1, total_games)
        
        # Role-specific win rates
        role_stats = {}
        for role, stats in self.role_performance.items():
            if stats['games'] > 0:
                role_stats[role] = {
                    'win_rate': stats['wins'] / stats['games'],
                    'games': stats['games'],
                    'avg_length': np.mean(self.game_length_by_role[role]) if self.game_length_by_role[role] else 0
                }
        
        return {
            'episodes': self.episodes,
            'win_rate': win_rate,
            'wins': self.wins,
            'losses': self.losses,
            'timeouts': self.timeouts,
            'avg_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'std_reward': np.std(self.episode_rewards) if self.episode_rewards else 0,
            'avg_cards_on_loss': np.mean(self.cards_remaining_on_loss) if self.cards_remaining_on_loss else 0,
            'avg_opponent_cards': np.mean(self.opponent_cards_remaining) if self.opponent_cards_remaining else 0,
            'role_performance': role_stats,
            'invalid_actions': self.invalid_actions,
            'max_consecutive_losses': self.max_consecutive_losses,
            'training_time': time.time() - self.start_time
        }

# Custom Callback for Doudizhu Training with Enhanced Metrics
class DoudizhuTrainingCallback(BaseCallback):
    def __init__(self, 
                 eval_freq: int = 1000,
                 save_freq: int = 5000,
                 checkpoint_dir: str = "checkpoints",
                 log_freq: int = 100,
                 verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.checkpoint_dir = checkpoint_dir
        self.log_freq = log_freq
        
        self.metrics = DoudizhuMetrics()
        self.episode_count = 0
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def _on_step(self) -> bool:
        # Track episode progress
        self.current_episode_length += 1
        self.current_episode_reward += self.locals.get('rewards', [0])[0]
        
        # Check if episode ended
        done = self.locals.get('dones', [False])[0]
        if done:
            info = self.locals.get('infos', [{}])[0]
            self.metrics.update(info, self.current_episode_reward, self.current_episode_length)
            self.episode_count += 1
            
            # Reset episode tracking
            self.current_episode_reward = 0
            self.current_episode_length = 0
            
            # Periodic logging
            if self.episode_count % self.log_freq == 0:
                self._log_progress()
            
            # Periodic evaluation
            if self.episode_count % self.eval_freq == 0:
                self._evaluate_performance()
            
            # Periodic checkpointing
            if self.episode_count % self.save_freq == 0:
                self._save_checkpoint()
        
        return True
    
    # Log Curr Training Progress
    def _log_progress(self):
        summary = self.metrics.get_summary()
        
        print(f"\nEpisode {self.episode_count} Progress ---")
        print(f"Win Rate: {summary['win_rate']:.3f}")
        print(f"Avg Episode Length: {summary['avg_episode_length']:.1f}")
        print(f"Avg Reward: {summary['avg_reward']:.2f}")
        print(f"Avg Cards on Loss: {summary['avg_cards_on_loss']:.1f}")
        
        # Role-specific performance
        for role, stats in summary['role_performance'].items():
            print(f"{role.capitalize()}: {stats['win_rate']:.3f} ({stats['games']} games)")
        
        # Log to tensorboard
        # LOGGING:
        #   - WR, avg ep len, invalid actions, Farmer/LL WR, Farmer/LL avg ep len 
        if hasattr(self, 'logger') and self.logger:
            self.logger.record('train/win_rate', summary['win_rate'])
            self.logger.record('train/avg_episode_length', summary['avg_episode_length'])
            self.logger.record('train/invalid_actions', summary['invalid_actions'])
            
            for role, stats in summary['role_performance'].items():
                self.logger.record(f'train/win_rate_{role}', stats['win_rate'])
                self.logger.record(f'train/avg_length_{role}', stats['avg_length'])

    # Run Eval Episodes
    def _evaluate_performance(self):
        if self.verbose > 0:
            summary = self.metrics.get_summary()
            print(f"\nEvaluation at Episode {self.episode_count}")
            print(f"Overall Win Rate: {summary['win_rate']:.3f}")
            print(f"Recent Performance: {summary['role_performance']}")
    
    # Save Model Checkpoint
    def _save_checkpoint(self):
        checkpoint_path = os.path.join(self.checkpoint_dir, f"model_episode_{self.episode_count}")
        self.model.save(checkpoint_path)
        
        # Save metrics
        metrics_path = os.path.join(self.checkpoint_dir, f"metrics_episode_{self.episode_count}.json")
        with open(metrics_path, 'w') as f:
            json.dump(convert_for_json(self.metrics.get_summary()), f, indent=2)
        
        if self.verbose > 0:
            print(f"Checkpoint saved: {checkpoint_path}")


# Create PPO Model w Config
def create_ppo_model(env, config: PPOConfig, stage_name: str = "unknown") -> MaskablePPO:
    # Set up tensorboard logging
    log_dir = f"logs/tensorboard/{stage_name}_{int(time.time())}"
    
    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        policy_kwargs=config.policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir
    )
    
    return model

# Update Model's Tensorboard logging dir forea new stage -- since ea. stage using same PPO Model
def update_tensorboard_logging(model, stage_name: str):
    new_log_dir = f"logs/tensorboard/{stage_name}_{int(time.time())}"
    model.tensorboard_log = new_log_dir
    
    # Force creation of new logger
    if hasattr(model, '_logger'):
        model._logger = None
    return new_log_dir

# Eval Agent Performance
def evaluate_agent(model, env, num_episodes: int = 100) -> Dict[str, Any]:
    metrics = DoudizhuMetrics()
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0

        # DEBUG
        if obs.shape != (28,):
            print(f"Warning: Initial obs shape is {obs.shape}, expected (28,)")
            continue
        
        while True:
            # CHECK: Only act if it's RL Agent's turn
            # (A) OPP's TURN
            if env.game.state.curr_player != env.rl_player:
                # It's opponent's turn - let them play
                env._opponent_turn()
                
                # Check if game ended after opponent's turn
                if env.game.is_game_over():
                    # Get final info and break
                    final_info = {
                        "episode_step": env.episode_step,
                        "winner": env.game.get_game_result()['winner'],
                        "rl_agent_won": env.game.get_game_result()['winner'] == env.rl_player,
                        "rl_agent_role": env.game.get_player_role(env.rl_player),
                        "cards_remaining": sum(env.game.state.player1_vec),
                        "opponent_cards_remaining": sum(env.game.state.player2_vec),
                        "cards_played_this_episode": env.cards_played_this_episode,
                        "opponent_type": env.get_opponent_strength()
                    }
                    metrics.update(final_info, episode_reward, episode_length)
                    break
                
                # Update observation after opponent's turn
                obs = env._get_observation()
                if obs.shape != (28,):
                    print(f"Error: Observation shape {obs.shape} after opponent turn")
                    break
                continue
            

            if obs.shape != (28,):
                print(f"Error: Invalid observation shape {obs.shape} before RL action")
                break


            # (B) RL AGENT's TURN
            # Get action mask and predict
            action_mask = env.action_masks()
            if not isinstance(action_mask, np.ndarray) or action_mask.shape[0] != env.num_actions:
                print(f"Error: Invalid action mask shape {action_mask.shape if hasattr(action_mask, 'shape') else type(action_mask)}")
                break
            if not np.any(action_mask):
                print("Warning: No valid actions available")
                break
            
            # action, _ = model.predict(obs, action_masks=action_mask, deterministic=True)
            try:
                action, _ = model.predict(obs.reshape(1, -1), action_masks=action_mask.reshape(1, -1), deterministic=True)
                if isinstance(action, np.ndarray):
                    action = action[0]
            except Exception as e:
                print(f"Error during model prediction: {e}")
                print(f"Obs shape: {obs.shape}, Action mask shape: {action_mask.shape}")
                break
            
            obs, reward, terminated, truncated, step_info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                metrics.update(step_info, episode_reward, episode_length)
                break
    
    return metrics.get_summary()


# Get PPO Config for Specific Training Stage
def get_stage_config(stage: int) -> PPOConfig:
    base_config = PPOConfig()
    
    if stage == 1: # Random opponent
        base_config.ent_coef = 0.1 # high exploration
        base_config.learning_rate = 3e-4
    elif stage == 2: # Mixed opponents
        base_config.ent_coef = 0.05 # med exploration
        base_config.learning_rate = 2e-4
    elif stage == 3: # Greedy opponent
        base_config.ent_coef = 0.02 # low exploration
        base_config.learning_rate = 1e-4
        base_config.clip_range = 0.15
    elif stage == 4: # Self-play
        base_config.ent_coef = 0.01 # minimal exploration
        base_config.learning_rate = 5e-5
        base_config.clip_range = 0.1
    elif stage == 5: # Mixed self-play + rule-based
        base_config.ent_coef = 0.015 # a litte more exploring
        base_config.learning_rate = 3e-5 # slower for stability
        base_config.clip_range = 0.1
    
    return base_config


# Convert Problematic Types for JSON Serialization
def convert_for_json(obj):
    if isinstance(obj, dict):
        return {str(k): convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_for_json(item) for item in obj]
    elif hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj

