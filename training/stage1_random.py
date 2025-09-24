# training/stage1_random.py

# Stage 1: PPO Training v Random Opp
# 0-5k episodes
# Goal: Learn basic game rules and valid move sequences
# Target: 
# - >80% WR against Rand Opp



import sys
import os
import random
import numpy as np
import json
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.insert(0, project_root)

from doudizhu.doudizhu_env import DoudizhuEnv
from agents.random_agent import RandomAgent
from training.utils.training_utils import (
    create_ppo_model, 
    get_stage_config, 
    DoudizhuTrainingCallback,
    evaluate_agent,
    convert_for_json
)



def main():
    print("Starting Stage 1 Training: PPO v Random Opp")
    print("=" * 60)
    
    # Training parameters
    TOTAL_EPISODES = 5000
    EVALUATION_EPISODES = 100
    TARGET_WIN_RATE = 0.8
    SEED = 42
    
    random.seed(SEED)
    np.random.seed(SEED)
    
    # Create environment with random opponent
    print("Setting up environment...")
    opponent = RandomAgent(seed=SEED)
    
    # Use conservative shaping reward config
    reward_config = {
        'win_reward': 5.0,
        'loss_penalty': -5.0,
        'step_penalty': -0.001,
        'hand_reduction_bonus': 0.02,
        'invalid_action_penalty': -0.1,
        'endgame_urgency_bonus': 0.1,
        'role_bonus_landlord': 0.0,
        'role_bonus_farmer': 0.0
    }
    
    env = DoudizhuEnv(
        opponent_agent=opponent,
        verbose=False,
        seed=SEED,
        reward_config=reward_config
    )
    
    print(f"Environment created with opponent: {type(opponent).__name__}")
    print(f"Reward config: conservative_shaping")
    print(f"Action space size: {env.action_space.n}")
    print(f"Observation space: {env.observation_space.shape}")
    
    # Get Stage 1 PPO configuration
    config = get_stage_config(stage=1)
    print(f"\nPPO Configuration:")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Entropy Coefficient: {config.ent_coef}")
    print(f"  Network Architecture: {config.policy_kwargs['net_arch']}")
    
    # Create PPO model
    print("\nCreating PPO model...")
    model = create_ppo_model(env, config, "stage1_random")
    
    # Setup callback with checkpoint directory
    checkpoint_dir = "training/checkpoints/stage1"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    callback = DoudizhuTrainingCallback(
        eval_freq=500, # eval every 500 episodes
        save_freq=1000, # save every 1000 episodes
        checkpoint_dir=checkpoint_dir,
        log_freq=100, # log every 100 episodes
        verbose=1
    )
    
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Tensorboard logs: logs/tensorboard/stage1_random_*")
    
    # Calculate total timesteps (approximate)
    # Each episode is ~50-200 steps, so use conservative estimate
    estimated_steps_per_episode = 100
    total_timesteps = TOTAL_EPISODES * estimated_steps_per_episode
    
    print(f"\nStarting training...")
    print(f"Target episodes: {TOTAL_EPISODES}")
    print(f"Estimated timesteps: {total_timesteps}")
    print(f"Target win rate: {TARGET_WIN_RATE}")
    print("=" * 60)
    
    # Start training
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=10 # log to tensorboard every 10 updates
        )
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.1f} seconds")
        
    except KeyboardInterrupt:
        print(f"\nTraining interrupted by user")
        training_time = time.time() - start_time
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise
    
    # Final Eval
    print(f"\nRunning final evaluation ({EVALUATION_EPISODES} episodes)...")
    
    final_results = evaluate_agent(model, env, num_episodes=EVALUATION_EPISODES)
    
    print(f"\nSTAGE 1 FINAL RES")
    print("=" * 40)
    print(f"Overall Win Rate: {final_results['win_rate']:.3f}")
    print(f"Total Episodes: {final_results['episodes']}")
    print(f"Average Episode Length: {final_results['avg_episode_length']:.1f}")
    print(f"Average Reward: {final_results['avg_reward']:.2f}")
    print(f"Average Cards on Loss: {final_results['avg_cards_on_loss']:.1f}")
    print(f"Invalid Actions: {final_results['invalid_actions']}")
    print(f"Max Consecutive Losses: {final_results['max_consecutive_losses']}")
    
    print(f"\nRole-Specific Performance:")
    for role, stats in final_results['role_performance'].items():
        print(f"  {role.capitalize()}: {stats['win_rate']:.3f} ({stats['games']} games, avg length: {stats['avg_length']:.1f})")
    
    # Save final model and results
    final_model_path = os.path.join(checkpoint_dir, "final_model")
    model.save(final_model_path)
    
    final_results_path = os.path.join(checkpoint_dir, "final_results.json")
    with open(final_results_path, 'w') as f:
        json.dump(convert_for_json(final_results), f, indent=2)
    
    print(f"\nFinal model saved: {final_model_path}")
    print(f"Final results saved: {final_results_path}")
    
    # Check if ready for Stage 2
    if final_results['win_rate'] >= TARGET_WIN_RATE:
        print(f"\nSUCCESS! Win rate {final_results['win_rate']:.3f} exceeds target {TARGET_WIN_RATE}")
        print(f"Ready to proceed to Stage 2 (Mixed opponents)")
    else:
        print(f"\nWin rate {final_results['win_rate']:.3f} below target {TARGET_WIN_RATE}")
    
    # Cleanup
    env.close()
    
    print(f"\nStage 1 training session complete!")
    print(f"Training time: {training_time:.1f} seconds")
    
    return final_results


if __name__ == "__main__":
    # Quick environment check before starting
    try:
        print("Running pre-training compatibility check...")
        test_env = DoudizhuEnv(opponent_agent=RandomAgent(), verbose=False)
        obs, info = test_env.reset()
        action_mask = test_env.action_masks()
        valid_actions = [i for i, valid in enumerate(action_mask) if valid]
        
        if not valid_actions:
            raise RuntimeError("No valid actions available - check environment setup")
        
        action = random.choice(valid_actions)
        obs, reward, terminated, truncated, step_info = test_env.step(action)
        test_env.close()
        
        print("Env compatibility check passed")
        print(f"    Observation shape: {obs.shape}")
        print(f"    Action space: {test_env.action_space.n}")
        print(f"    Sample valid actions: {len(valid_actions)}")
        
    except Exception as e:
        print(f"Env check failed: {e}")
        print("Please fix env issues before starting training")
        exit(1)
    
    results = main()

