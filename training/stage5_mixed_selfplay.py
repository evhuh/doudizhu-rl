# training/stage5_mixed_selfplay.py

# Stage 5: Mixed Self-Play Training (80% self-play, 20% rule-based opponents)
# Goal: Maintain diversity while leveraging self-play strength
# 60k episodes

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


# Wrapper: Use a Trained PPO Model as Opp Agent (copied from stage4)
class PPOAgent:
    def __init__(self, model, env_reference=None, deterministic=True):
        self.model = model
        self.deterministic = deterministic
        self.env_reference = env_reference
        self.name = f"PPO_Agent_{id(self)}"
    
    def choose_action(self, legal_moves, game_state):
        try:
            observation = self.env_reference._get_observation()
            
            if observation.shape != (28,):
                print(f"PPOAgent error: Invalid observation shape {observation.shape}")
                return random.choice(legal_moves)
            
            action_mask = np.zeros(self.model.action_space.n, dtype=bool)
            move_to_index_map = {}
            
            for i, move in enumerate(legal_moves):
                action_idx = self.env_reference.move_to_action(move)
                if action_idx is not None and 0 <= action_idx < len(action_mask):
                    action_mask[action_idx] = True
                    move_to_index_map[action_idx] = i
            
            if not np.any(action_mask):
                print("PPOAgent error: No valid actions in mask")
                return random.choice(legal_moves)
            
            action, _ = self.model.predict(observation, action_masks=action_mask, deterministic=self.deterministic)
            
            if hasattr(action, 'item'):
                chosen_action_idx = int(action.item())
            elif np.isscalar(action):
                chosen_action_idx = int(action)
            elif hasattr(action, '__len__') and len(action) > 0:
                chosen_action_idx = int(action[0])
            else:
                print(f"PPOAgent error: Unexpected action format: {type(action)}, {action}")
                return random.choice(legal_moves)
            
            if chosen_action_idx in move_to_index_map:
                legal_move_idx = move_to_index_map[chosen_action_idx]
                return legal_moves[legal_move_idx]
            else:
                print(f"PPOAgent error: Could not map chosen action {chosen_action_idx} back to legal move")
                return random.choice(legal_moves)
                
        except Exception as e:
            print(f"PPOAgent error: {e}")
            return random.choice(legal_moves)
    
    def reset(self):
        pass


# Enhanced Opponent Manager: Self-Play + Rule-Based Mix
class MixedSelfPlayOpponentManager:
    def __init__(self, stage2_model_path, stage3_model_path, stage4_model_path, current_model, main_env, 
                 update_frequency=2000, selfplay_ratio=0.8):
        self.stage2_model_path = stage2_model_path
        self.stage3_model_path = stage3_model_path
        self.stage4_model_path = stage4_model_path
        self.current_model = current_model
        self.main_env = main_env
        self.update_frequency = update_frequency
        self.episodes_since_update = 0
        self.selfplay_ratio = selfplay_ratio
        
        # Calculate weights based on desired ratio
        # Self-play: current + historical models
        selfplay_weight = selfplay_ratio
        rule_based_weight = 1.0 - selfplay_ratio
        
        # Self-play distribution (current gets more weight)
        current_weight = selfplay_weight * 0.5    # 35% of total (50% of 70%)
        stage4_weight = selfplay_weight * 0.25    # 17.5% of total
        stage3_weight = selfplay_weight * 0.15    # 10.5% of total  
        stage2_weight = selfplay_weight * 0.1     # 7% of total
        
        # Rule-based distribution (30% total, evenly split)
        greedy_weight = rule_based_weight * 0.4 # 12% of total
        conservative_weight = rule_based_weight * 0.4 # 12% of total
        random_weight = rule_based_weight * 0.2 # 6% of total
        
        self.pool_weights = [current_weight, stage4_weight, stage3_weight, stage2_weight, 
                           greedy_weight, conservative_weight, random_weight]
        
        # Load all opponents
        self.stage2_agent = None
        self.stage3_agent = None
        self.stage4_agent = None
        self.current_agent = None
        
        # Rule-based agents
        self.greedy_agent = GreedyAgent()
        self.conservative_agent = ConservativeAgent()
        self.random_agent = RandomAgent()
        
        self._load_historical_models()
        self._update_current_model()
        
        # Pool statistics
        self.selections = {
            'current': 0, 'stage4': 0, 'stage3': 0, 'stage2': 0,
            'greedy': 0, 'conservative': 0, 'random': 0
        }
        
        print(f"Mixed Self-Play Opponent Pool Initialized:")
        print(f"  Self-Play Ratio: {selfplay_ratio:.1%}")
        print(f"  Current Model: {current_weight:.1%}")
        print(f"  Stage 4 Model: {stage4_weight:.1%}")
        print(f"  Stage 3 Model: {stage3_weight:.1%}")
        print(f"  Stage 2 Model: {stage2_weight:.1%}")
        print(f"  Greedy Agent: {greedy_weight:.1%}")
        print(f"  Conservative Agent: {conservative_weight:.1%}")
        print(f"  Random Agent: {random_weight:.1%}")
    
    def _load_historical_models(self):
        print("Loading historical models for mixed opponent pool...")
        
        # Load Stage 2
        try:
            stage2_model = MaskablePPO.load(self.stage2_model_path, env=self.main_env)
            self.stage2_agent = PPOAgent(stage2_model, env_reference=self.main_env, deterministic=True)
            print("Stage 2 model loaded")
        except Exception as e:
            print(f"Failed to load Stage 2 model: {e}")
            self.stage2_agent = GreedyAgent()
        
        # Load Stage 3
        try:
            stage3_model = MaskablePPO.load(self.stage3_model_path, env=self.main_env)
            self.stage3_agent = PPOAgent(stage3_model, env_reference=self.main_env, deterministic=True)
            print("Stage 3 model loaded")
        except Exception as e:
            print(f"Failed to load Stage 3 model: {e}")
            self.stage3_agent = GreedyAgent()
        
        # Load Stage 4
        try:
            stage4_model = MaskablePPO.load(self.stage4_model_path, env=self.main_env)
            self.stage4_agent = PPOAgent(stage4_model, env_reference=self.main_env, deterministic=True)
            print("Stage 4 model loaded")
        except Exception as e:
            print(f"Failed to load Stage 4 model: {e}")
            self.stage4_agent = GreedyAgent()
    
    def _update_current_model(self):
        self.current_agent = PPOAgent(self.current_model, env_reference=self.main_env, deterministic=False)
    
    def select_opponent(self):
        choices = ['current', 'stage4', 'stage3', 'stage2', 'greedy', 'conservative', 'random']
        choice = np.random.choice(choices, p=self.pool_weights)
        self.selections[choice] += 1
        
        opponent_map = {
            'current': self.current_agent,
            'stage4': self.stage4_agent,
            'stage3': self.stage3_agent,
            'stage2': self.stage2_agent,
            'greedy': self.greedy_agent,
            'conservative': self.conservative_agent,
            'random': self.random_agent
        }
        
        return opponent_map[choice]
    
    def episode_finished(self):
        self.episodes_since_update += 1
        
        if self.episodes_since_update >= self.update_frequency:
            print(f"\nUpdating opponent pool (every {self.update_frequency} episodes)")
            self._update_current_model()
            self.episodes_since_update = 0
            
            # Print selection statistics
            total = sum(self.selections.values())
            if total > 0:
                print(f"Recent opponent selections:")
                selfplay_count = sum([self.selections[k] for k in ['current', 'stage4', 'stage3', 'stage2']])
                rulebased_count = sum([self.selections[k] for k in ['greedy', 'conservative', 'random']])
                
                print(f"  Self-Play: {selfplay_count}/{total} ({100*selfplay_count/total:.1f}%)")
                print(f"  Rule-Based: {rulebased_count}/{total} ({100*rulebased_count/total:.1f}%)")
                
                for opponent, count in self.selections.items():
                    print(f"    {opponent}: {count} ({100*count/total:.1f}%)")
                
                # Reset statistics
                self.selections = {k: 0 for k in self.selections.keys()}
    
    def get_stats(self):
        return {
            'episodes_since_update': self.episodes_since_update,
            'update_frequency': self.update_frequency,
            'selfplay_ratio': self.selfplay_ratio,
            'pool_weights': self.pool_weights,
            'selections': self.selections.copy()
        }


def main():
    print("Stage 5 Training: Mixed Self-Play + Rule-Based (70/30)")
    print("=" * 60)
    
    # Training parameters
    TOTAL_EPISODES = 60000
    EVALUATION_EPISODES = 150
    OPPONENT_UPDATE_FREQ = 2000
    SELFPLAY_RATIO = 0.8
    TARGET_WIN_RATE = 0.60
    SEED = 42
    
    random.seed(SEED)
    np.random.seed(SEED)
    
    # Create environment
    print("Setting up mixed self-play environment...")
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
    
    initial_opponent = GreedyAgent()
    env = DoudizhuEnv(
        opponent_agent=initial_opponent,
        verbose=False,
        seed=SEED,
        reward_config=reward_config
    )
    
    # Load Stage 4 model as starting point
    stage4_model_path = "training/checkpoints/stage4/final_model"
    print(f"Loading Stage 4 model from: {stage4_model_path}")
    
    try:
        model = MaskablePPO.load(stage4_model_path, env=env)
        print("Successfully loaded Stage 4 model")
        update_tensorboard_logging(model, "stage5_mixed_selfplay")
    except Exception as e:
        print(f"Failed to load Stage 4 model: {e}")
        return None
    
    # Get Stage 5 configuration (for display/logging purposes)
    config = get_stage_config(stage=5)
    print(f"\nStage 5 PPO Configuration:")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Entropy Coefficient: {config.ent_coef}")
    print(f"  Mixed Training: {SELFPLAY_RATIO:.1%} self-play, {1-SELFPLAY_RATIO:.1%} rule-based")
    
    # Setup mixed opponent manager
    stage2_model_path = "training/checkpoints/stage2/final_model"
    stage3_model_path = "training/checkpoints/stage3/final_model"
    
    opponent_manager = MixedSelfPlayOpponentManager(
        stage2_model_path=stage2_model_path,
        stage3_model_path=stage3_model_path,
        stage4_model_path=stage4_model_path,
        current_model=model,
        main_env=env,
        update_frequency=OPPONENT_UPDATE_FREQ,
        selfplay_ratio=SELFPLAY_RATIO
    )
    
    # Setup callback
    checkpoint_dir = "training/checkpoints/stage5"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    class MixedSelfPlayCallback(DoudizhuTrainingCallback):
        def __init__(self, opponent_manager, **kwargs):
            super().__init__(**kwargs)
            self.opponent_manager = opponent_manager
            self.episode_count = 0
        
        def _on_step(self) -> bool:
            if hasattr(self.locals, 'dones') and self.locals['dones'] is not None:
                if any(self.locals['dones']):
                    self.episode_count += 1
                    self.opponent_manager.episode_finished()
                    new_opponent = self.opponent_manager.select_opponent()
                    self.model.env.set_opponent(new_opponent)
            
            return super()._on_step()
        
        def _log_progress(self):
            super()._log_progress()
            if hasattr(self, 'episode_count'):
                stats = self.opponent_manager.get_stats()
                print(f"Episodes: {self.episode_count}, Next pool update in: {stats['update_frequency'] - stats['episodes_since_update']}")
    
    callback = MixedSelfPlayCallback(
        opponent_manager=opponent_manager,
        eval_freq=1500,
        save_freq=3000,
        checkpoint_dir=checkpoint_dir,
        log_freq=300,
        verbose=1
    )
    
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Tensorboard logs: logs/tensorboard/stage5_mixed_selfplay_*")
    
    # Calculate timesteps
    estimated_steps_per_episode = 100
    total_timesteps = TOTAL_EPISODES * estimated_steps_per_episode
    
    print(f"\nStarting Stage 5 mixed self-play training...")
    print(f"Target episodes: {TOTAL_EPISODES}")
    print(f"Self-play ratio: {SELFPLAY_RATIO:.1%}")
    print(f"Estimated timesteps: {total_timesteps}")
    print("=" * 60)
    
    # Training
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=10,
            reset_num_timesteps=False
        )
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.1f} seconds")
        
    except KeyboardInterrupt:
        print(f"\nTraining interrupted by user")
        training_time = time.time() - start_time
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise
    
    # Final evaluation (same as Stage 4)
    print(f"\nRunning final evaluation against all opponent types...")
    
    # Rule-based opponents
    print(f"Evaluating vs Random agent ({EVALUATION_EPISODES} episodes)...")
    random_opponent = RandomAgent()
    eval_env = DoudizhuEnv(opponent_agent=random_opponent, verbose=False, seed=SEED, reward_config=reward_config)
    random_results = evaluate_agent(model, eval_env, num_episodes=EVALUATION_EPISODES)

    print(f"Evaluating vs Conservative agent ({EVALUATION_EPISODES} episodes)...")
    conservative_opponent = ConservativeAgent()
    eval_env.set_opponent(conservative_opponent)
    conservative_results = evaluate_agent(model, eval_env, num_episodes=EVALUATION_EPISODES)
    
    print(f"Evaluating vs Greedy agent ({EVALUATION_EPISODES} episodes)...")
    greedy_opponent = GreedyAgent()
    eval_env.set_opponent(greedy_opponent)
    greedy_results = evaluate_agent(model, eval_env, num_episodes=EVALUATION_EPISODES)

    # Historical models
    print(f"Evaluating vs Stage 2 model ({EVALUATION_EPISODES} episodes)...")
    eval_env.set_opponent(opponent_manager.stage2_agent)
    stage2_results = evaluate_agent(model, eval_env, num_episodes=EVALUATION_EPISODES)

    print(f"Evaluating vs Stage 3 model ({EVALUATION_EPISODES} episodes)...")
    eval_env.set_opponent(opponent_manager.stage3_agent)
    stage3_results = evaluate_agent(model, eval_env, num_episodes=EVALUATION_EPISODES)

    print(f"Evaluating vs Stage 4 model ({EVALUATION_EPISODES} episodes)...")
    eval_env.set_opponent(opponent_manager.stage4_agent)
    stage4_results = evaluate_agent(model, eval_env, num_episodes=EVALUATION_EPISODES)

    # Results
    print(f"\nSTAGE 5 FINAL RESULTS")
    print("=" * 60)
    print(f"VS RULE-BASED OPPONENTS:")
    print(f"  Greedy:      {greedy_results['win_rate']:.3f} (avg len: {greedy_results['avg_episode_length']:.1f})")
    print(f"  Conservative: {conservative_results['win_rate']:.3f} (avg len: {conservative_results['avg_episode_length']:.1f})")
    print(f"  Random:      {random_results['win_rate']:.3f} (avg len: {random_results['avg_episode_length']:.1f})")
    
    print(f"\nVS HISTORICAL SELF-VERSIONS:")
    print(f"  Stage 4 Model: {stage4_results['win_rate']:.3f} (avg len: {stage4_results['avg_episode_length']:.1f})")
    print(f"  Stage 3 Model: {stage3_results['win_rate']:.3f} (avg len: {stage3_results['avg_episode_length']:.1f})")
    print(f"  Stage 2 Model: {stage2_results['win_rate']:.3f} (avg len: {stage2_results['avg_episode_length']:.1f})")
    
    print(f"\nOVERALL PERFORMANCE:")
    print(f"  Invalid Actions: {greedy_results['invalid_actions']}")
    print(f"  Cards on Loss: {greedy_results['avg_cards_on_loss']:.1f}")
    
    # Role-specific performance
    print(f"\nRole-Specific Performance (vs Greedy):")
    for role, stats in greedy_results['role_performance'].items():
        print(f"  {role.capitalize()}: {stats['win_rate']:.3f} ({stats['games']} games)")
    
    # Save results
    final_model_path = os.path.join(checkpoint_dir, "final_model")
    model.save(final_model_path)
    
    results_summary = {
        'stage': 5,
        'training_type': 'mixed_selfplay',
        'selfplay_ratio': SELFPLAY_RATIO,
        'training_episodes': TOTAL_EPISODES,
        'training_time': training_time,
        'opponent_pool_stats': opponent_manager.get_stats(),
        'vs_greedy': greedy_results,
        'vs_conservative': conservative_results,
        'vs_random': random_results,
        'vs_stage4_model': stage4_results,
        'vs_stage3_model': stage3_results,
        'vs_stage2_model': stage2_results
    }
    
    final_results_path = os.path.join(checkpoint_dir, "final_results.json")
    with open(final_results_path, 'w') as f:
        json.dump(convert_for_json(results_summary), f, indent=2)
    
    print(f"\nFinal model saved: {final_model_path}")
    print(f"Final results saved: {final_results_path}")
    
    # Performance evolution comparison
    print(f"\nCOMPLETE PERFORMANCE EVOLUTION:")
    print("=" * 40)
    print(f"Stage 5: Random {random_results['win_rate']:.0%} | Conservative {conservative_results['win_rate']:.0%} | Greedy {greedy_results['win_rate']:.0%}")
    print(f"         vs Stage4: {stage4_results['win_rate']:.0%} | vs Stage3: {stage3_results['win_rate']:.0%}")
    
    # Mixed training insights
    print(f"\nMIXED TRAINING INSIGHTS:")
    stats = opponent_manager.get_stats()
    total_selections = sum(opponent_manager.selections.values())
    if total_selections > 0:
        selfplay_selections = sum([opponent_manager.selections[k] for k in ['current', 'stage4', 'stage3', 'stage2']])
        rulebased_selections = sum([opponent_manager.selections[k] for k in ['greedy', 'conservative', 'random']])
        
        print(f"Actual training distribution:")
        print(f"  Self-Play: {100*selfplay_selections/total_selections:.1f}% (target: {100*SELFPLAY_RATIO:.1f}%)")
        print(f"  Rule-Based: {100*rulebased_selections/total_selections:.1f}% (target: {100*(1-SELFPLAY_RATIO):.1f}%)")
    
    # Success evaluation
    avg_performance = np.mean([
        greedy_results['win_rate'], 
        conservative_results['win_rate'],
        stage4_results['win_rate']
    ])
    
    print(f"\nAverage performance vs challenging opponents: {avg_performance:.1%}")
    
    if avg_performance >= TARGET_WIN_RATE:
        print("SUCCESS! Mixed self-play training achieved target performance!")
        print("Model combines self-play strength with rule-based robustness!")
    else:
        print(f"Performance at {avg_performance:.1%} - consider adjusting mix ratio or extended training")
    
    # Cleanup
    env.close()
    eval_env.close()
    
    print(f"\nModel is ready for deployment or advanced training continuation")
    print(f"Stage 5 mixed self-play training complete!")
    print(f"Total training time: {training_time/3600:.1f} hours")
    
    return results_summary


if __name__ == "__main__":
    # Pre-training checks
    try:
        print("Running pre-training compatibility check...")
        
        required_models = [
            ("Stage 2", "training/checkpoints/stage2/final_model"),
            ("Stage 3", "training/checkpoints/stage3/final_model"),
            ("Stage 4", "training/checkpoints/stage4/final_model")
        ]
        
        missing_models = []
        for name, path in required_models:
            if not os.path.exists(path + ".zip") and not os.path.exists(path):
                missing_models.append(name)
        
        if missing_models:
            raise FileNotFoundError(f"Missing models: {', '.join(missing_models)}. Complete previous stages first.")
        
        print("Mixed self-play compatibility check passed")
        print("  All required model stages found")
        print("  Ready for 70% self-play / 30% rule-based training")
        
    except Exception as e:
        print(f"Pre-training check failed: {e}")
        exit(1)
    
    results = main()

