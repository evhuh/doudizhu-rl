# training/stage2_mixed.py

# Stage 2: PPO Training v Mixed Opps (Random : Conservative | 7:3)
# 10k episodes
# Goal: Learn strat basics against diff play styles
# Target: 
# - >60% WR against Conservative Opp


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



# Creat Opp that Rand Switches btw Rand and Conservative Agents
def create_mixed_opponent(random_prob=0.7, seed=None):
    if random.random() < random_prob:
        return RandomAgent(seed=seed)
    else:
        return ConservativeAgent()

def main():
    print("Stage 2 Training: PPO vs Mixed Opponents (Random + Conservative)")
    print("=" * 60)
    
    # Training Params
    TOTAL_EPISODES = 10000 # 10k additional eps
    EVALUATION_EPISODES = 100
    TARGET_WIN_RATE = 0.6  # Target vs Conservative agent
    RANDOM_OPPONENT_PROB = 0.7 # 70% Random, 30% Conservative
    SEED = 42
    
    random.seed(SEED)
    np.random.seed(SEED)
    
    # Create Environment FIRST (same config as Stage 1)
    print("Setting up environment...")
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
    
    initial_opponent = RandomAgent(seed=SEED)
    env = DoudizhuEnv(
        opponent_agent=initial_opponent,
        verbose=False,
        seed=SEED,
        reward_config=reward_config
    )
    
    print(f"Environment created with mixed opponents:")
    print(f"  Random agent probability: {RANDOM_OPPONENT_PROB}")
    print(f"  Conservative agent probability: {1-RANDOM_OPPONENT_PROB}")
    
    # Load Stage 1 model WITH environment
    stage1_model_path = "training/checkpoints/stage1/final_model"
    print(f"Loading Stage 1 model from: {stage1_model_path}")
    
    try:
        # Load the pre-trained model with environment
        model = MaskablePPO.load(stage1_model_path, env=env)
        print(f"Successfully loaded Stage 1 model")
        update_tensorboard_logging(model, "stage2_mixed")
    except Exception as e:
        print(f"Failed to load Stage 1 model: {e}")
        print("Make sure Stage 1 training completed successfully")
        return None
    
    # Get Stage 2 configuration (for display/logging purposes)
    config = get_stage_config(stage=2)
    print(f"\nStage 2 PPO Configuration:")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Entropy Coefficient: {config.ent_coef} (reduced exploration)")
    
    # (A): Keep Stage 1 hyperparameters (USEFUL for stability)
    print(f"Using Stage 1 model hyperparameters for continuity")
    
    # (B): TO CHANGE and use Stage 2 config (model with diff hyperparams), create new model with loaded weights
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
    
    # Setup callback with Stage 2 checkpoint directory
    checkpoint_dir = "training/checkpoints/stage2"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Custom callback that switches opponents dynamically
    class MixedOpponentCallback(DoudizhuTrainingCallback):
        def __init__(self, random_prob=0.7, **kwargs):
            super().__init__(**kwargs)
            self.random_prob = random_prob
            self.opponent_switches = 0
            self.episode_count = 0
        
        def _on_step(self) -> bool:
            # Check if ep just ended using done signal
            if hasattr(self.locals, 'dones') and self.locals['dones'] is not None:
                if any(self.locals['dones']): # ep ended
                    self.episode_count += 1
                    # Switch opponent for next ep
                    new_opponent = create_mixed_opponent(self.random_prob, SEED)
                    self.model.env.set_opponent(new_opponent)
                    self.opponent_switches += 1
            
            return super()._on_step()
        
        def _log_progress(self):
            super()._log_progress()
            if hasattr(self, 'episode_count'):
                print(f"Episodes completed: {self.episode_count}, Opponent switches: {self.opponent_switches}")
    
    callback = MixedOpponentCallback(
        random_prob=RANDOM_OPPONENT_PROB,
        eval_freq=1000, # eval every 1000 eps  
        save_freq=2000, # save every 2000 eps
        checkpoint_dir=checkpoint_dir,
        log_freq=200, # log every 200 eps
        verbose=1
    )
    
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Tensorboard logs: logs/tensorboard/stage2_mixed_*")
    
    # Calc total timesteps
    estimated_steps_per_episode = 100
    total_timesteps = TOTAL_EPISODES * estimated_steps_per_episode
    
    print(f"\nStarting Stage 2 training...")
    print(f"Target episodes: {TOTAL_EPISODES}")
    print(f"Estimated timesteps: {total_timesteps}")
    print(f"Target win rate vs Conservative: {TARGET_WIN_RATE}")
    print("=" * 60)
    
    # Start training
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=10,
            reset_num_timesteps=False # cont timestep counting from Stage 1
        )
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.1f} seconds")
        
    except KeyboardInterrupt:
        print(f"\nTraining interrupted by user")
        training_time = time.time() - start_time
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise
    
    # (1) Final eval against Conservative Agent (specifically)
    print(f"\nRunning final evaluation vs Conservative Agent ({EVALUATION_EPISODES} episodes)...")
    
    # Create env with only Conservative Opp for eval
    conservative_opponent = ConservativeAgent()
    eval_env = DoudizhuEnv(
        opponent_agent=conservative_opponent,
        verbose=False,
        seed=SEED,
        reward_config=reward_config
    )
    
    conservative_results = evaluate_agent(model, eval_env, num_episodes=EVALUATION_EPISODES)
    
    # (2) Final eval against Random (for comparison)
    print(f"Running comparison evaluation vs Random agent ({EVALUATION_EPISODES} episodes)...")
    random_opponent = RandomAgent()
    eval_env.set_opponent(random_opponent)
    random_results = evaluate_agent(model, eval_env, num_episodes=EVALUATION_EPISODES)
    
    print(f"\nSTAGE 2 FINAL RESULTS")
    print("=" * 50)
    print(f"VS CONSERVATIVE AGENT:")
    print(f"  Win Rate: {conservative_results['win_rate']:.3f}")
    print(f"  Avg Episode Length: {conservative_results['avg_episode_length']:.1f}")
    print(f"  Avg Cards on Loss: {conservative_results['avg_cards_on_loss']:.1f}")
    print(f"  Invalid Actions: {conservative_results['invalid_actions']}")
    
    print(f"\nVS RANDOM AGENT (comparison):")
    print(f"  Win Rate: {random_results['win_rate']:.3f}")
    print(f"  Avg Episode Length: {random_results['avg_episode_length']:.1f}")
    
    print(f"\nRole-Specific Performance vs Conservative:")
    for role, stats in conservative_results['role_performance'].items():
        print(f"  {role.capitalize()}: {stats['win_rate']:.3f} ({stats['games']} games)")
    
    # Save final model and res
    final_model_path = os.path.join(checkpoint_dir, "final_model")
    model.save(final_model_path)
    
    # Save both eval res
    results_summary = {
        'stage': 2,
        'training_episodes': TOTAL_EPISODES,
        'training_time': training_time,
        'vs_conservative': conservative_results,
        'vs_random': random_results,
        'opponent_mix': {
            'random_prob': RANDOM_OPPONENT_PROB,
            'conservative_prob': 1 - RANDOM_OPPONENT_PROB
        }
    }
    
    final_results_path = os.path.join(checkpoint_dir, "final_results.json")
    with open(final_results_path, 'w') as f:
        json.dump(convert_for_json(results_summary), f, indent=2)
    
    print(f"\nFinal model saved: {final_model_path}")
    print(f"Final results saved: {final_results_path}")
    
    # Check if ready for Stage 3
    if conservative_results['win_rate'] >= TARGET_WIN_RATE:
        print(f"\nSUCCESS! Win rate vs Conservative {conservative_results['win_rate']:.3f} exceeds target {TARGET_WIN_RATE}")
        print(f"Ready to proceed to Stage 3 (Greedy opponent)")
    else:
        print(f"\nWin rate vs Conservative {conservative_results['win_rate']:.3f} below target {TARGET_WIN_RATE}")

    # Performance comparison
    print(f"\nPerformance Comparison:")
    print(f"  Stage 2 (vs Random): {random_results['win_rate']:.3f}")
    print(f"  Stage 2 (vs Conservative): {conservative_results['win_rate']:.3f}")
    
    # Cleanup
    env.close()
    eval_env.close()
    
    print(f"\nStage 2 training session complete!")
    
    return results_summary


if __name__ == "__main__":
    # Pre-training checks
    try:
        print("Running pre-training compatibility check...")
        
        # Check if Stage 1 model exists (try both .zip and without)
        stage1_path = "training/checkpoints/stage1/final_model"
        if not os.path.exists(stage1_path + ".zip") and not os.path.exists(stage1_path):
            raise FileNotFoundError("Stage 1 model not found. Complete Stage 1 training first.")
        
        # Test mixed opponent creation
        test_random = create_mixed_opponent(random_prob=1.0) # should be Random
        test_conservative = create_mixed_opponent(random_prob=0.0) # should be Conservative
        
        # Test environment with set_opponent method
        test_env = DoudizhuEnv(opponent_agent=RandomAgent(), verbose=False)
        test_env.set_opponent(ConservativeAgent()) # ts will fail if method doesn't exist
        test_env.close()
        
        print("Environment compatibility check passed")
        print("Mixed opponent system working")
        
    except Exception as e:
        print(f"Pre-training check failed: {e}")
        if "set_opponent" in str(e):
            print("\nNeed to add set_opponent() method to DoudizhuEnv:")
            print("def set_opponent(self, new_opponent):")
            print("    self.opponent_agent = new_opponent")
        exit(1)
    
    results = main()

