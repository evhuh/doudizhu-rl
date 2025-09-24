# training/reward_tuning.py

# Reward Tuning Suite for RL Training
# Test diff reward structs w short training sessions:
# - Baseline
# - Efficiency Best
# - High Risk High Reward
# - Conservative, Gentle Reward Shaping



import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Any
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.insert(0, project_root)

from doudizhu.doudizhu_env import DoudizhuEnv



# Config for Rwward Struct
@dataclass
class RewardConfig:
    name: str
    win_reward: float = 10.0
    loss_penalty: float = -10.0
    step_penalty: float = -0.01
    invalid_action_penalty: float = -1.0
    card_play_bonus: float = 0.0
    hand_reduction_bonus: float = 0.0
    game_length_penalty: float = 0.0
    consecutive_pass_penalty: float = 0.0
    description: str = ""


# Wrapper to Modify Env Rewards Based on Diff Strats
class RewardShaper:
    def __init__(self, env, config: RewardConfig):
        self.env = env
        self.config = config
        self.last_hand_size = None
        self.consecutive_passes = 0
        self.episode_length = 0
        
    def reset(self):
        obs, info = self.env.reset()
        self.last_hand_size = None
        self.consecutive_passes = 0
        self.episode_length = 0
        
        # Try to get initial hand size from observation or info
        if hasattr(self.env, 'get_hand_size'):
            self.last_hand_size = self.env.get_hand_size()
        elif 'hand_size' in info:
            self.last_hand_size = info['hand_size']
            
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Apply reward shaping
        shaped_reward = self._shape_reward(action, reward, obs, info, terminated, truncated)
        
        self.episode_length += 1
        return obs, shaped_reward, terminated, truncated, info
    
    def _shape_reward(self, action, original_reward, obs, info, terminated, truncated):
        """Apply reward shaping based on configuration"""
        shaped_reward = original_reward
        
        # Win/loss rewards (replace original terminal rewards)
        if terminated:
            if original_reward > 0:  # Win
                shaped_reward = self.config.win_reward
            else:  # Loss
                shaped_reward = self.config.loss_penalty
        else:
            # Step penalty
            shaped_reward += self.config.step_penalty
            
            # Invalid action penalty
            if 'invalid_action' in info and info['invalid_action']:
                shaped_reward += self.config.invalid_action_penalty
            
            # Card play bonus
            if 'cards_played' in info and info['cards_played'] > 0:
                shaped_reward += self.config.card_play_bonus * info['cards_played']
            
            # Hand reduction bonus
            current_hand_size = None
            if hasattr(self.env, 'get_hand_size'):
                current_hand_size = self.env.get_hand_size()
            elif 'hand_size' in info:
                current_hand_size = info['hand_size']
                
            if current_hand_size is not None and self.last_hand_size is not None:
                hand_reduction = self.last_hand_size - current_hand_size
                if hand_reduction > 0:
                    shaped_reward += self.config.hand_reduction_bonus * hand_reduction
                self.last_hand_size = current_hand_size
            
            # Game length penalty
            if self.config.game_length_penalty != 0:
                shaped_reward += self.config.game_length_penalty * self.episode_length / 100.0
            
            # Consecutive pass penalty
            if 'action_type' in info:
                if info['action_type'] == 'pass':
                    self.consecutive_passes += 1
                else:
                    self.consecutive_passes = 0
                    
                if self.consecutive_passes > 2:
                    shaped_reward += self.config.consecutive_pass_penalty
        
        return shaped_reward
    
    # Delegate Other Attributes to the Wrapped Env
    def __getattr__(self, name):
        return getattr(self.env, name)


# Run Experiments w Diff Reward Structs
class RewardTuningExperiment:
    def __init__(self, env_class=DoudizhuEnv):
        self.env_class = env_class
        self.reward_configs = self._create_reward_configs()
        
    # Diff Reward Configs to Test
    def _create_reward_configs(self) -> List[RewardConfig]:
        configs = [
            RewardConfig(
                name="baseline",
                description="Standard win/loss rewards only"
            ),
            
            RewardConfig(
                name="efficiency_focus",
                win_reward=15.0,
                loss_penalty=-5.0,
                step_penalty=-0.05,
                game_length_penalty=-0.01,
                consecutive_pass_penalty=-0.1,
                description="Rewards efficient, quick games"
            ),
            
            RewardConfig(
                name="sparse_high_stakes",
                win_reward=50.0,
                loss_penalty=-50.0,
                step_penalty=0.0,
                description="High stakes, sparse rewards"
            ),
            
            RewardConfig(
                name="conservative_shaping",
                win_reward=5.0,
                loss_penalty=-5.0,
                step_penalty=-0.001,
                hand_reduction_bonus=0.02,
                invalid_action_penalty=-0.1,
                description="Gentle shaping for stable learning"
            )
        ]
        
        return configs
    
    # Run Training Exp w Specific Reward Config
    def run_experiment(self, config: RewardConfig, num_episodes: int = 1000) -> Dict[str, Any]:
        print(f"\nTesting reward config: {config.name}")
        print(f"   Description: {config.description}")
        
        # Create shaped environment
        base_env = self.env_class(verbose=False)
        env = RewardShaper(base_env, config)
        
        # Initialize tracking
        results = {
            'config_name': config.name,
            'episodes_completed': 0,
            'wins': 0,
            'losses': 0,
            'timeouts': 0,
            'episode_lengths': [],
            'episode_rewards': [],
            'win_rates_over_time': [],
            'action_counts': defaultdict(int),
            'move_type_counts': defaultdict(int),
            'invalid_actions': 0,
            'avg_cards_remaining': [],
            'role_performance': defaultdict(lambda: {'wins': 0, 'games': 0})
        }
        
        # Running avgs for tracking progress
        win_rate_window = deque(maxlen=100)
        
        print(f"   Running {num_episodes} episodes...")
        start_time = time.time()
        
        for episode in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_actions = []
            
            while True:
                # Get valid actions
                valid_actions = env.legal_actions()
                if not valid_actions:
                    action = env.action_space.sample()  # Fallback
                else:
                    action = np.random.choice(valid_actions)  # Random policy for testing
                
                episode_actions.append(action)
                obs, reward, terminated, truncated, step_info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                # Track invalid actions
                if step_info.get('invalid_action', False):
                    results['invalid_actions'] += 1
                
                if terminated or truncated:
                    break
                    
                if episode_length > 500:  # Safety break
                    break
            
            # Record episode results
            results['episodes_completed'] += 1
            results['episode_lengths'].append(episode_length)
            results['episode_rewards'].append(episode_reward)
            
            # Track action distribution
            for action in episode_actions:
                results['action_counts'][action] += 1
                try:
                    move = env.action_to_move(action)
                    results['move_type_counts'][move.combo_type] += 1
                except:
                    pass
            
            # Track game outcome
            if terminated:
                if step_info.get('rl_agent_won', False):
                    results['wins'] += 1
                    win_rate_window.append(1)
                else:
                    results['losses'] += 1
                    win_rate_window.append(0)
                    
                # Track role performance
                role = step_info.get('rl_agent_role', 'unknown')
                results['role_performance'][role]['games'] += 1
                if step_info.get('rl_agent_won', False):
                    results['role_performance'][role]['wins'] += 1
                    
                # Track cards remaining
                cards_remaining = step_info.get('rl_cards_remaining', 0)
                results['avg_cards_remaining'].append(cards_remaining)
                
            elif truncated:
                results['timeouts'] += 1
                win_rate_window.append(0)
            
            # Track win rate over time
            if len(win_rate_window) >= 10:
                current_win_rate = sum(win_rate_window) / len(win_rate_window)
                results['win_rates_over_time'].append(current_win_rate)
            
            # Progress reporting
            if episode > 0 and episode % max(1, num_episodes // 10) == 0:
                current_win_rate = results['wins'] / max(1, results['wins'] + results['losses'])
                elapsed_time = time.time() - start_time
                eps_per_sec = episode / elapsed_time
                print(f"     Episode {episode}/{num_episodes} - Win Rate: {current_win_rate:.3f} - {eps_per_sec:.1f} eps/sec")
        
        # Calculate final statistics
        total_games = results['wins'] + results['losses']
        results['final_win_rate'] = results['wins'] / max(1, total_games)
        results['avg_episode_length'] = np.mean(results['episode_lengths']) if results['episode_lengths'] else 0
        results['avg_episode_reward'] = np.mean(results['episode_rewards']) if results['episode_rewards'] else 0
        results['std_episode_reward'] = np.std(results['episode_rewards']) if results['episode_rewards'] else 0
        results['avg_cards_remaining'] = np.mean(results['avg_cards_remaining']) if results['avg_cards_remaining'] else 0
        results['total_time'] = time.time() - start_time
        
        # Action diversity metrics
        total_actions = sum(results['action_counts'].values())
        unique_actions = len(results['action_counts'])
        results['action_diversity'] = unique_actions / max(1, env.action_space.n)
        results['action_entropy'] = self._calculate_entropy(list(results['action_counts'].values()))
        
        print(f"Completed! Win Rate: {results['final_win_rate']:.3f}, Avg Length: {results['avg_episode_length']:.1f}")
        return results
    
    # Calc Shannon Entropy of Action Distribution -- EE 2020 :D
    def _calculate_entropy(self, counts):
        if not counts or sum(counts) == 0:
            return 0.0
        
        total = sum(counts)
        probs = [c / total for c in counts if c > 0]
        entropy = -sum(p * np.log2(p) for p in probs)
        return entropy
    
    # Run Exp w All Reward Configs
    def run_all_experiments(self, num_episodes: int = 1000) -> Dict[str, Any]:
        print("Starting Reward Tuning Experiments")
        print("=" * 60)
        
        all_results = {}
        comparison_data = {
            'configs': [],
            'win_rates': [],
            'episode_lengths': [],
            'action_diversities': [],
            'reward_means': [],
            'reward_stds': []
        }
        
        for config in self.reward_configs:
            results = self.run_experiment(config, num_episodes)
            all_results[config.name] = results
            
            # Collect comparison data
            comparison_data['configs'].append(config.name)
            comparison_data['win_rates'].append(results['final_win_rate'])
            comparison_data['episode_lengths'].append(results['avg_episode_length'])
            comparison_data['action_diversities'].append(results['action_diversity'])
            comparison_data['reward_means'].append(results['avg_episode_reward'])
            comparison_data['reward_stds'].append(results['std_episode_reward'])
        
        # Generate comparison report
        self._generate_comparison_report(all_results, comparison_data)
        
        # Create visualizations
        self._create_experiment_plots(all_results, comparison_data)
        
        # Save results
        with open('reward_tuning_results.json', 'w') as f:
            json.dump(convert_for_json(all_results), f, indent=2)
        
        print(f"\nAll experiments completed! Results saved to 'reward_tuning_results.json'")
        return all_results
    
    # Generate Comparison Report of All Configs
    def _generate_comparison_report(self, all_results, comparison_data):
        print("\nREWARD CONFIGURATION COMPARISON")
        print("=" * 60)
        
        # Sort by win rate
        sorted_configs = sorted(zip(comparison_data['configs'], 
                                  comparison_data['win_rates'],
                                  comparison_data['episode_lengths'],
                                  comparison_data['action_diversities'],
                                  comparison_data['reward_means']),
                               key=lambda x: x[1], reverse=True)
        
        print(f"{'Rank':<4} {'Config':<18} {'Win Rate':<10} {'Avg Length':<12} {'Diversity':<10} {'Avg Reward':<12}")
        print("-" * 80)
        
        for i, (config, win_rate, length, diversity, reward) in enumerate(sorted_configs, 1):
            print(f"{i:<4} {config:<18} {win_rate:<10.3f} {length:<12.1f} {diversity:<10.3f} {reward:<12.2f}")
        
        # Best performing config details
        best_config = sorted_configs[0][0]
        best_results = all_results[best_config]
        
        print(f"\nBEST PERFORMING CONFIG: {best_config}")
        print("-" * 40)
        print(f"Win Rate: {best_results['final_win_rate']:.3f}")
        print(f"Average Episode Length: {best_results['avg_episode_length']:.1f}")
        print(f"Action Diversity: {best_results['action_diversity']:.3f}")
        print(f"Action Entropy: {best_results['action_entropy']:.2f}")
        print(f"Invalid Actions: {best_results['invalid_actions']}")
        
        # Role performance breakdown
        if best_results['role_performance']:
            print(f"\nRole Performance:")
            for role, perf in best_results['role_performance'].items():
                if perf['games'] > 0:
                    role_win_rate = perf['wins'] / perf['games']
                    print(f"  {role}: {role_win_rate:.3f} ({perf['wins']}/{perf['games']})")
    
    # Create Visualization Plots
    def _create_experiment_plots(self, all_results, comparison_data):
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # 1. Win Rates Comparison
            axes[0, 0].bar(comparison_data['configs'], comparison_data['win_rates'])
            axes[0, 0].set_title('Win Rates by Configuration')
            axes[0, 0].set_ylabel('Win Rate')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. Episode Lengths
            axes[0, 1].bar(comparison_data['configs'], comparison_data['episode_lengths'])
            axes[0, 1].set_title('Average Episode Length')
            axes[0, 1].set_ylabel('Episode Length')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # 3. Action Diversity
            axes[0, 2].bar(comparison_data['configs'], comparison_data['action_diversities'])
            axes[0, 2].set_title('Action Diversity')
            axes[0, 2].set_ylabel('Diversity Score')
            axes[0, 2].tick_params(axis='x', rotation=45)
            
            # 4. Win Rate Over Time (best config)
            best_config = max(all_results.keys(), key=lambda k: all_results[k]['final_win_rate'])
            best_results = all_results[best_config]
            if best_results['win_rates_over_time']:
                axes[1, 0].plot(best_results['win_rates_over_time'])
                axes[1, 0].set_title(f'Win Rate Over Time ({best_config})')
                axes[1, 0].set_xlabel('Episode (x100)')
                axes[1, 0].set_ylabel('Win Rate')
            
            # 5. Reward Distribution (best config)
            if best_results['episode_rewards']:
                axes[1, 1].hist(best_results['episode_rewards'], bins=30, alpha=0.7)
                axes[1, 1].set_title(f'Reward Distribution ({best_config})')
                axes[1, 1].set_xlabel('Episode Reward')
                axes[1, 1].set_ylabel('Frequency')
            
            # 6. Move Type Distribution (best config)
            if best_results['move_type_counts']:
                move_types = list(best_results['move_type_counts'].keys())
                move_counts = list(best_results['move_type_counts'].values())
                axes[1, 2].pie(move_counts, labels=move_types, autopct='%1.1f%%')
                axes[1, 2].set_title(f'Move Type Distribution ({best_config})')
            
            plt.tight_layout()
            plt.savefig('reward_tuning_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("Plots saved as 'reward_tuning_analysis.png'")
            
        except ImportError:
            print("[-] Matplotlib not available - skipping plot generation")


# Convert Problematic Types for JSON Serialization
def convert_for_json(obj):
    if isinstance(obj, dict):
        # Convert defaultdict and handle int64 keys
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


def main():
    print("DoudiZhu Reward Tuning Suite")
    print("=" * 50)
    
    # 1. Quick compatibility check first
    print("Running quick PPO compatibility check...")
    try:
        env = DoudizhuEnv(verbose=False)
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        print("Basic environment functionality confirmed")
    except Exception as e:
        print(f"[X] Environment issue detected: {e}")
        return
    
    # 2. Run reward tuning experiments
    tuner = RewardTuningExperiment()
    
    # 3. Run quick test (100 episodes each) for initial screening
    print("\nRunning initial screening (100 episodes per config)...")
    quick_results = tuner.run_all_experiments(num_episodes=100)
    
    # 4. Get user input for full run
    print(f"\nQuick screening complete. Best config: {max(quick_results.keys(), key=lambda k: quick_results[k]['final_win_rate'])}")
    
    full_run = input("\nRun full experiments (1000 episodes each)? [y/N]: ").lower().strip()
    if full_run == 'y':
        print("\nRunning full experiments...")
        full_results = tuner.run_all_experiments(num_episodes=1000)
        
        # Suggest best config for PPO training
        best_config_name = max(full_results.keys(), key=lambda k: full_results[k]['final_win_rate'])
        best_config = next(c for c in tuner.reward_configs if c.name == best_config_name)
        
        print(f"\nRECOMMENDATION FOR PPO TRAINING:")
        print(f"Use reward configuration: '{best_config.name}'")
        print(f"Description: {best_config.description}")
        print(f"Expected win rate: {full_results[best_config_name]['final_win_rate']:.3f}")
    
    print("\nReward tuning experiments completed!")

if __name__ == "__main__":
    main()

