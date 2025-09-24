#!/usr/bin/env python3

# Unit tests for PPO Compatibility w DoudiZhu Env
# - gym interface compliance, action space, RL training loop



import numpy as np
from typing import Dict, Any
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.insert(0, project_root)
    
from doudizhu.doudizhu_env import DoudizhuEnv



class PPOCompatibilityTester:
    def __init__(self, env_class=DoudizhuEnv):
        self.env_class = env_class
        self.test_results = {}
    
    # Test Basic Gym Interface Compliance
    def test_gym_interface(self) -> Dict[str, bool]:
        print("Testing Gym Interface Compliance...")
        results = {}
        
        try:
            env = self.env_class()
            
            # Test 1: Environment creation
            results['env_creation'] = True
            print("Environment creation: PASSED")
            
            # Test 2: Reset functionality
            obs, info = env.reset()
            results['reset_returns_tuple'] = isinstance(obs, (np.ndarray, dict)) and isinstance(info, dict)
            print(f"Reset returns (obs, info): {'PASSED' if results['reset_returns_tuple'] else 'FAILED'}")
            
            # Test 3: Step functionality
            action = env.action_space.sample()
            step_result = env.step(action)
            results['step_returns_5_tuple'] = len(step_result) == 5
            obs, reward, terminated, truncated, info = step_result
            print(f"Step returns 5-tuple: {'PASSED' if results['step_returns_5_tuple'] else 'FAILED'}")
            
            # Test 4: Action space
            results['has_action_space'] = hasattr(env, 'action_space')
            results['action_space_sample'] = hasattr(env.action_space, 'sample')
            results['action_space_contains'] = hasattr(env.action_space, 'contains')
            print(f"Action space interface: {'PASSED' if all([results['has_action_space'], results['action_space_sample'], results['action_space_contains']]) else 'FAILED'}")
            
            # Test 5: Observation space
            results['has_observation_space'] = hasattr(env, 'observation_space')
            results['obs_in_space'] = env.observation_space.contains(obs)
            print(f"Observation space interface: {'PASSED' if results['has_observation_space'] and results['obs_in_space'] else 'FAILED'}")
            
            # Test 6: Reward type
            results['reward_is_numeric'] = isinstance(reward, (int, float, np.integer, np.floating))
            print(f"Reward is numeric: {'PASSED' if results['reward_is_numeric'] else 'FAILED'}")
            
            # Test 7: Termination flags
            results['termination_flags_bool'] = isinstance(terminated, bool) and isinstance(truncated, bool)
            print(f"Termination flags are boolean: {'PASSED' if results['termination_flags_bool'] else 'FAILED'}")
            
        except Exception as e:
            print(f"[X] Gym interface test failed with error: {e}")
            results['exception'] = str(e)
            
        return results
    
    # Test Action Space Propoerties for PPO Compatibility
    def test_action_space_properties(self) -> Dict[str, Any]:
        print("\nTesting Action Space Properties...")
        results = {}
        
        env = self.env_class()
        env.reset()
        
        # Test action space type
        action_space = env.action_space
        results['action_space_type'] = type(action_space).__name__
        
        if hasattr(action_space, 'n'):
            # Discrete action space
            results['is_discrete'] = True
            results['action_space_size'] = action_space.n
            print(f"Discrete action space with {action_space.n} actions")
            
            # Test action sampling
            actions = [action_space.sample() for _ in range(100)]
            results['action_range_valid'] = all(0 <= a < action_space.n for a in actions)
            results['action_distribution'] = np.bincount(actions, minlength=action_space.n).tolist()
            print(f"Action sampling: {'PASSED' if results['action_range_valid'] else 'FAILED'}")
            
        else:
            # Continuous action space (less common for card games)
            results['is_discrete'] = False
            print("[-] Continuous action space detected (unusual for card games)")
            
        return results
    
    # Test Observation Space Properties
    def test_observation_space_properties(self) -> Dict[str, Any]:
        print("\nTesting Observation Space Properties...")
        results = {}
        
        env = self.env_class()
        obs, _ = env.reset()
        
        # Test observation properties
        results['obs_type'] = type(obs).__name__
        results['obs_shape'] = obs.shape if hasattr(obs, 'shape') else 'no_shape'
        results['obs_dtype'] = str(obs.dtype) if hasattr(obs, 'dtype') else str(type(obs))
        
        if isinstance(obs, np.ndarray):
            results['obs_min'] = float(obs.min())
            results['obs_max'] = float(obs.max())
            results['obs_mean'] = float(obs.mean())
            results['has_nan'] = bool(np.any(np.isnan(obs)))
            results['has_inf'] = bool(np.any(np.isinf(obs)))
            
            print(f"Observation shape: {obs.shape}")
            print(f"Observation dtype: {obs.dtype}")
            print(f"Value range: [{obs.min():.3f}, {obs.max():.3f}]")
            print(f"NaN/Inf check: {'FAILED' if results['has_nan'] or results['has_inf'] else 'PASSED'}")
            
        return results
    
    # Test Episode Mechanics and Consistency
    def test_episode_mechanics(self, num_episodes: int = 10) -> Dict[str, Any]:
        print(f"\nTesting Episode Mechanics ({num_episodes} episodes)...")
        results = {
            'episodes_completed': 0,
            'episode_lengths': [],
            'total_rewards': [],
            'termination_reasons': defaultdict(int),
            'errors': []
        }
        
        env = self.env_class()
        
        for episode in range(num_episodes):
            try:
                obs, info = env.reset()
                episode_length = 0
                total_reward = 0
                
                while True:
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    episode_length += 1
                    total_reward += reward
                    
                    if terminated or truncated:
                        results['termination_reasons']['terminated' if terminated else 'truncated'] += 1
                        break
                        
                    if episode_length > 1000: # safety check
                        results['termination_reasons']['max_steps'] += 1
                        break
                
                results['episodes_completed'] += 1
                results['episode_lengths'].append(episode_length)
                results['total_rewards'].append(total_reward)
                
                if episode % max(1, num_episodes // 4) == 0:
                    print(f"  Episode {episode + 1}/{num_episodes}: Length={episode_length}, Reward={total_reward:.3f}")
                    
            except Exception as e:
                results['errors'].append(f"Episode {episode}: {str(e)}")
                print(f"[X] Episode {episode} failed: {e}")
        
        # Calculate statistics
        if results['episode_lengths']:
            results['avg_episode_length'] = np.mean(results['episode_lengths'])
            results['std_episode_length'] = np.std(results['episode_lengths'])
            results['avg_total_reward'] = np.mean(results['total_rewards'])
            results['std_total_reward'] = np.std(results['total_rewards'])
            
            print(f"Completed episodes: {results['episodes_completed']}/{num_episodes}")
            print(f"Average episode length: {results['avg_episode_length']:.1f} ± {results['std_episode_length']:.1f}")
            print(f"Average total reward: {results['avg_total_reward']:.3f} ± {results['std_total_reward']:.3f}")
        
        return results
    
    # Env Supports Action Masking
    def test_action_masking(self) -> Dict[str, Any]:
        print("\nTesting Action Masking...")
        results = {}
        
        env = self.env_class()
        obs, info = env.reset()
        
        # Check if action mask is provided in info
        results['has_action_mask_in_info'] = 'action_mask' in info
        
        if results['has_action_mask_in_info']:
            mask = info['action_mask']
            results['mask_type'] = type(mask).__name__
            results['mask_shape'] = mask.shape if hasattr(mask, 'shape') else 'no_shape'
            results['mask_dtype'] = str(mask.dtype) if hasattr(mask, 'dtype') else str(type(mask))
            
            if isinstance(mask, np.ndarray):
                results['mask_is_binary'] = np.all((mask == 0) | (mask == 1))
                results['valid_actions_count'] = int(np.sum(mask))
                results['total_actions'] = len(mask)
                
                print(f"Action mask found in info")
                print(f"Mask shape: {mask.shape}")
                print(f"Valid actions: {results['valid_actions_count']}/{results['total_actions']}")
                print(f"Binary mask: {'PASSED' if results['mask_is_binary'] else 'FAILED'}")
        
        # Check if observation includes action mask
        if isinstance(obs, dict):
            results['has_action_mask_in_obs'] = 'action_mask' in obs
            print(f"Action mask in observation: {'YES' if results['has_action_mask_in_obs'] else 'NO'}")
        else:
            results['has_action_mask_in_obs'] = False
        
        return results
    
    # Anal Reward Struct for RL Training
    def test_reward_structure(self, num_episodes: int = 50) -> Dict[str, Any]:
        print(f"\nTesting Reward Structure ({num_episodes} episodes)...")
        results = {
            'rewards': [],
            'episode_rewards': [],
            'reward_types': defaultdict(int),
            'sparse_rewards': 0,
            'dense_rewards': 0
        }
        
        env = self.env_class()
        
        for episode in range(num_episodes):
            obs, info = env.reset()
            episode_rewards = []
            
            while True:
                # Use environment's action masking if available
                if hasattr(env, 'legal_actions'):
                    legal_actions = env.legal_actions()
                    if legal_actions:
                        action = np.random.choice(legal_actions)
                    else:
                        action = env.action_space.sample()
                else:
                    action = env.action_space.sample()
                    
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_rewards.append(reward)
                results['rewards'].append(reward)
                
                # Categorize reward
                if reward == 0:
                    results['sparse_rewards'] += 1
                else:
                    results['dense_rewards'] += 1
                    
                if abs(reward) < 0.001:
                    results['reward_types']['zero'] += 1
                elif reward > 0:
                    results['reward_types']['positive'] += 1
                else:
                    results['reward_types']['negative'] += 1
                
                if terminated or truncated:
                    break
            
            results['episode_rewards'].append(sum(episode_rewards))
        
        # Calculate statistics
        all_rewards = np.array(results['rewards'])
        episode_rewards = np.array(results['episode_rewards'])
        
        results['reward_stats'] = {
            'min_step_reward': float(all_rewards.min()),
            'max_step_reward': float(all_rewards.max()),
            'mean_step_reward': float(all_rewards.mean()),
            'std_step_reward': float(all_rewards.std()),
            'min_episode_reward': float(episode_rewards.min()),
            'max_episode_reward': float(episode_rewards.max()),
            'mean_episode_reward': float(episode_rewards.mean()),
            'std_episode_reward': float(episode_rewards.std()),
        }
        
        results['reward_sparsity'] = results['sparse_rewards'] / len(results['rewards'])
        
        print(f"Step rewards range: [{all_rewards.min():.3f}, {all_rewards.max():.3f}]")
        print(f"Episode rewards range: [{episode_rewards.min():.3f}, {episode_rewards.max():.3f}]")
        print(f"Reward sparsity: {results['reward_sparsity']:.1%} zero rewards")
        print(f"Reward distribution: {dict(results['reward_types'])}")
        
        return results
    

    # Generate Compatibility Report
    def generate_compatibility_report(self) -> str:
        print("\nGenerating Compatibility Report...")
        
        # Run all tests
        gym_results = self.test_gym_interface()
        action_results = self.test_action_space_properties()
        obs_results = self.test_observation_space_properties()
        episode_results = self.test_episode_mechanics()
        mask_results = self.test_action_masking()
        reward_results = self.test_reward_structure()
        
        # Store all results
        self.test_results = {
            'gym_interface': gym_results,
            'action_space': action_results,
            'observation_space': obs_results,
            'episode_mechanics': episode_results,
            'action_masking': mask_results,
            'reward_structure': reward_results
        }
        
        # Generate report
        report = self._create_report_text()
        
        # Save results to JSON
        with open('ppo_unit_test.json', 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        print("Results saved to 'ppo_unit_test.json'")
        return report
    
    def _create_report_text(self) -> str:
        """Create formatted report text"""
        report = "PPO Compatibility Report for DouDiZhu Environment\n"
        
        # Overall compatibility assessment
        gym_passed = sum(v for v in self.test_results['gym_interface'].values() if isinstance(v, bool))
        total_gym_tests = len([v for v in self.test_results['gym_interface'].values() if isinstance(v, bool)])
        
        compatibility_score = gym_passed / total_gym_tests if total_gym_tests > 0 else 0
        
        report += f"Overall Compatibility Score: {compatibility_score:.1%}\n\n"
        
        # Add specific recommendations
        recommendations = []
        
        if self.test_results['action_masking'].get('has_action_mask_in_info'):
            recommendations.append("Use masked PPO implementation to handle invalid actions")
        else:
            recommendations.append("[-] Consider adding action masking to prevent invalid moves")
        
        if self.test_results['reward_structure'].get('reward_sparsity', 1) > 0.8:
            recommendations.append("[-] Very sparse rewards - consider reward shaping or curriculum learning")
        
        if self.test_results['episode_mechanics'].get('avg_episode_length', 0) > 500:
            recommendations.append("[-] Long episodes - consider using GAE (Generalized Advantage Estimation)")
        
        if not recommendations:
            recommendations.append("Environment appears well-suited for standard PPO training")
        
        for rec in recommendations:
            report += f"- {rec}\n"
        
        return report


def main():   
    tester = PPOCompatibilityTester()
    report = tester.generate_compatibility_report()
    print(report)
    
    # Visualize
    try:
        create_test_plots(tester.test_results)
        print("\nTest plots saved as 'ppo_unit_test.png'")
    except ImportError:
        print("\nMatplotlib not available - skipping plot generation")

def create_test_plots(test_results):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Episode lengths
    episode_lengths = test_results['episode_mechanics'].get('episode_lengths', [])
    if episode_lengths:
        axes[0, 0].hist(episode_lengths, bins=20, alpha=0.7)
        axes[0, 0].set_title('Episode Length Distribution')
        axes[0, 0].set_xlabel('Episode Length')
        axes[0, 0].set_ylabel('Frequency')
    
    # Episode rewards
    episode_rewards = test_results['reward_structure'].get('episode_rewards', [])
    if episode_rewards:
        axes[0, 1].hist(episode_rewards, bins=20, alpha=0.7)
        axes[0, 1].set_title('Episode Reward Distribution')
        axes[0, 1].set_xlabel('Total Episode Reward')
        axes[0, 1].set_ylabel('Frequency')
    
    # Step rewards
    step_rewards = test_results['reward_structure'].get('rewards', [])
    if step_rewards:
        axes[1, 0].hist(step_rewards, bins=20, alpha=0.7)
        axes[1, 0].set_title('Step Reward Distribution')
        axes[1, 0].set_xlabel('Step Reward')
        axes[1, 0].set_ylabel('Frequency')
    
    # Action distribution (if available)
    action_dist = test_results['action_space'].get('action_distribution', [])
    if action_dist and len(action_dist) <= 50:  # Only plot if reasonable number of actions
        axes[1, 1].bar(range(len(action_dist)), action_dist)
        axes[1, 1].set_title('Action Sampling Distribution')
        axes[1, 1].set_xlabel('Action ID')
        axes[1, 1].set_ylabel('Sample Count')
    else:
        axes[1, 1].text(0.5, 0.5, 'Action distribution\nnot visualizable\n(too many actions)', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    plt.savefig('ppo_unit_test.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()

