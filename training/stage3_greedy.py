# training/stage3_greedy.py

# Stage 3: PPO Traiing v Greedy Opp
# 10k episodes
# Goal: Learn advanced strat against agressive play style
# Target: 
# - >60% WR against Greedy Opp



import sys
import os
import random
import numpy as np
import json
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.insert(0, project_root)

from doudizhu.doudizhu_env import DoudizhuEnv
from agents.greedy_agent import GreedyAgent
from agents.random_agent import RandomAgent
from agents.conservative_agent import ConservativeAgent
from training.utils.training_utils import (
    create_ppo_model, 
    update_tensorboard_logging,
    get_stage_config, 
    DoudizhuTrainingCallback,
    evaluate_agent,
    convert_for_json
)
from sb3_contrib import MaskablePPO




def main():
    print("Stage 3 Training: PPO vs Greedy Opponent")
    print("=" * 60)
    
    # Training Params
    TOTAL_EPISODES = 10000 # 10k additional episodes
    EVALUATION_EPISODES = 100
    TARGET_WIN_RATE = 0.6 # Target vs Greedy agent
    SEED = 42
    
    random.seed(SEED)
    np.random.seed(SEED)
    
    # Create Environment FIRST (same config as prev stages)
    print("Setting up environment with Greedy opponent...")
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
    
    greedy_opponent = GreedyAgent()
    env = DoudizhuEnv(
        opponent_agent=greedy_opponent,
        verbose=False,
        seed=SEED,
        reward_config=reward_config
    )
    
    print(f"Environment created with Greedy opponent")
    print(f"Action space size: {env.action_space.n}")
    print(f"Observation space: {env.observation_space.shape}")
    
    # Load Stage 2 model w/ Env
    stage2_model_path = "training/checkpoints/stage2/final_model"
    print(f"Loading Stage 2 model from: {stage2_model_path}")
    
    try:
        # Load the pre-trained model with Env
        model = MaskablePPO.load(stage2_model_path, env=env)
        print(f"Successfully loaded Stage 2 model")
        update_tensorboard_logging(model, "stage3_greedy")
    except Exception as e:
        print(f"Failed to load Stage 2 model: {e}")
        print("Make sure Stage 2 training completed successfully")
        return None
    
    # Get Stage 3 config (DEBUGGING)
    config = get_stage_config(stage=3)
    print(f"\nStage 3 PPO Configuration:")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Entropy Coefficient: {config.ent_coef} (further reduced exploration)")

     # (A): Keep Stage 2 hyperparameters (USEFUL for stability)
    print(f"Using Stage 2 model hyperparameters for continuity")
    
    # (B): TO CHANGE and use Stage 3 config (model with diff hyperparams), create new model with loaded weights
    """
    print("Creating new model with Stage 2 hyperparameters...")
    old_model = model
    model = create_ppo_model(env, config, "stage2_mixed")
    
    # Copy the trained weights from Stage 1 model
    model.policy.load_state_dict(old_model.policy.state_dict())
    if hasattr(model, 'value_net') and hasattr(old_model, 'value_net'):
        model.value_net.load_state_dict(old_model.value_net.state_dict())
    
    del old_model # free memory
    """
    
    # Setup callback with Stage 3 checkpt dir
    checkpoint_dir = "training/checkpoints/stage3"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Std callback (always Greedy Agent)
    callback = DoudizhuTrainingCallback(
        eval_freq=1000, # eval every 1000 eps  
        save_freq=2000, # save every 2000 eps
        checkpoint_dir=checkpoint_dir,
        log_freq=200, # log every 200 eps
        verbose=1
    )
    
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Tensorboard logs: logs/tensorboard/stage3_greedy_*")
    
    # Calculate total timesteps
    estimated_steps_per_episode = 100
    total_timesteps = TOTAL_EPISODES * estimated_steps_per_episode
    
    print(f"\nStarting Stage 3 training...")
    print(f"Target episodes: {TOTAL_EPISODES}")
    print(f"Estimated timesteps: {total_timesteps}")
    print(f"Target win rate vs Greedy: {TARGET_WIN_RATE}")
    print("=" * 60)
    
    # Start training
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=10,
            reset_num_timesteps=False # continue timestep counting from previous stages
        )
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.1f} seconds")
        
    except KeyboardInterrupt:
        print(f"\nTraining interrupted by user")
        training_time = time.time() - start_time
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise
    
    # Final evaluation against Greedy agent
    print(f"\nRunning final evaluation vs Greedy agent ({EVALUATION_EPISODES} episodes)...")
    
    greedy_results = evaluate_agent(model, env, num_episodes=EVALUATION_EPISODES)
    
    # Also evaluate against previous opponents for comparison
    print(f"Running comparison evaluation vs Conservative agent ({EVALUATION_EPISODES} episodes)...")
    conservative_opponent = ConservativeAgent()
    eval_env = DoudizhuEnv(
        opponent_agent=conservative_opponent,
        verbose=False,
        seed=SEED,
        reward_config=reward_config
    )
    conservative_results = evaluate_agent(model, eval_env, num_episodes=EVALUATION_EPISODES)
    
    print(f"Running comparison evaluation vs Random agent ({EVALUATION_EPISODES} episodes)...")
    random_opponent = RandomAgent()
    eval_env.set_opponent(random_opponent)
    random_results = evaluate_agent(model, eval_env, num_episodes=EVALUATION_EPISODES)
    
    print(f"\nSTAGE 3 FINAL RESULTS")
    print("=" * 50)
    print(f"VS GREEDY AGENT:")
    print(f"  Win Rate: {greedy_results['win_rate']:.3f}")
    print(f"  Avg Episode Length: {greedy_results['avg_episode_length']:.1f}")
    print(f"  Avg Cards on Loss: {greedy_results['avg_cards_on_loss']:.1f}")
    print(f"  Invalid Actions: {greedy_results['invalid_actions']}")
    
    print(f"\nVS CONSERVATIVE AGENT (comparison):")
    print(f"  Win Rate: {conservative_results['win_rate']:.3f}")
    print(f"  Avg Episode Length: {conservative_results['avg_episode_length']:.1f}")
    
    print(f"\nVS RANDOM AGENT (comparison):")
    print(f"  Win Rate: {random_results['win_rate']:.3f}")
    print(f"  Avg Episode Length: {random_results['avg_episode_length']:.1f}")
    
    print(f"\nRole-Specific Performance vs Greedy:")
    for role, stats in greedy_results['role_performance'].items():
        print(f"  {role.capitalize()}: {stats['win_rate']:.3f} ({stats['games']} games)")
    
    # Save final model and results
    final_model_path = os.path.join(checkpoint_dir, "final_model")
    model.save(final_model_path)
    
    # Save comprehensive results
    results_summary = {
        'stage': 3,
        'training_episodes': TOTAL_EPISODES,
        'training_time': training_time,
        'vs_greedy': greedy_results,
        'vs_conservative': conservative_results,
        'vs_random': random_results,
        'opponent_type': 'greedy_exclusive'
    }
    
    final_results_path = os.path.join(checkpoint_dir, "final_results.json")
    with open(final_results_path, 'w') as f:
        json.dump(convert_for_json(results_summary), f, indent=2)
    
    print(f"\nFinal model saved: {final_model_path}")
    print(f"Final results saved: {final_results_path}")
    
    # Check if ready for Stage 4
    if greedy_results['win_rate'] >= TARGET_WIN_RATE:
        print(f"\nSUCCESS! Win rate vs Greedy {greedy_results['win_rate']:.3f} exceeds target {TARGET_WIN_RATE}")
        print(f"Ready to proceed to Stage 4 (Self-play)")
    else:
        print(f"\nWin rate vs Greedy {greedy_results['win_rate']:.3f} below target {TARGET_WIN_RATE}")
    
    print(f"\nPerformance:")
    print(f"  Stage 3 (vs Random): {random_results['win_rate']:.1%} | (vs Conservative): {conservative_results['win_rate']:.1%} | (vs Greedy): {greedy_results['win_rate']:.1%}")
    
    # Strategy insights
    print(f"\nStrategy Insights:")
    if greedy_results['avg_episode_length'] < conservative_results['avg_episode_length']:
        print(f"  - Playing faster vs Greedy ({greedy_results['avg_episode_length']:.1f} vs {conservative_results['avg_episode_length']:.1f} steps)")
        print(f"  - Adapting to aggressive play style")
    
    if greedy_results['role_performance']['landlord']['win_rate'] != greedy_results['role_performance']['farmer']['win_rate']:
        landlord_wr = greedy_results['role_performance']['landlord']['win_rate']
        farmer_wr = greedy_results['role_performance']['farmer']['win_rate']
        if landlord_wr > farmer_wr:
            print(f"  - Stronger as Landlord vs Greedy ({landlord_wr:.1%} vs {farmer_wr:.1%})")
        else:
            print(f"  - Stronger as Farmer vs Greedy ({farmer_wr:.1%} vs {landlord_wr:.1%})")
    
    # Cleanup
    env.close()
    eval_env.close()
    
    print(f"\nStage 3 training session complete!")
    
    return results_summary


if __name__ == "__main__":
    # Pre-training checks
    try:
        print("Running pre-training compatibility check...")
        
        # Check if Stage 2 model exists
        stage2_path = "training/checkpoints/stage2/final_model"
        if not os.path.exists(stage2_path + ".zip") and not os.path.exists(stage2_path):
            raise FileNotFoundError("Stage 2 model not found. Complete Stage 2 training first.")
        
        # Test Greedy agent creation
        test_greedy = GreedyAgent()
        print(f"Greedy agent created: {type(test_greedy).__name__}")
        
        # Test environment with Greedy opponent
        test_env = DoudizhuEnv(opponent_agent=test_greedy, verbose=False)
        obs, info = test_env.reset()
        action_mask = test_env.action_masks()
        valid_actions = [i for i, valid in enumerate(action_mask) if valid]
        
        if not valid_actions:
            raise RuntimeError("No valid actions available with Greedy opponent")
        
        # Test one step
        action = random.choice(valid_actions)
        obs, reward, terminated, truncated, step_info = test_env.step(action)
        test_env.close()
        
        print("Environment compatibility check passed")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Action space: {test_env.action_space.n}")
        print(f"  Sample valid actions: {len(valid_actions)}")
        
    except Exception as e:
        print(f"Pre-training check failed: {e}")
        print("Please ensure:")
        print("  - Stage 2 training completed successfully")
        print("  - GreedyAgent is properly implemented")
        print("  - DoudizhuEnv has set_opponent() method")
        exit(1)
    
    results = main()

