# training/stage4_selfplay.py

# Stage 4: PPO Self-PLay Training (against previous versions of itself: current_model, stage2_checkpoint, stage3_checkpoint) (update ppol every 2k eps w latest model)
# 15k episodes
# Goal: Advanced strats through self-play
# Target: ...80


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



# Wrapper: Use a Trained PPO Model as Opp Agent
class PPOAgent:
    def __init__(self, model, env_reference=None, deterministic=True):
        self.model = model
        self.deterministic = deterministic
        self.env_reference = env_reference # ref env for observation extraction
        self.name = f"PPO_Agent_{id(self)}"
    
    # Choose Action Using the PPO Model (match w Env interface) 
    def choose_action(self, legal_moves, game_state):
        try:
            # Get observation
            observation = self.env_reference._get_observation()
            
            if observation.shape != (28,):
                print(f"PPOAgent error: Invalid observation shape {observation.shape}")
                return random.choice(legal_moves) # ret Move object, not index
            
            # Create action mask for the full action space
            action_mask = np.zeros(self.model.action_space.n, dtype=bool)
            move_to_index_map = {} # action_space index -> legal_moves index
            
            for i, move in enumerate(legal_moves):
                action_idx = self.env_reference.move_to_action(move)
                if action_idx is not None and 0 <= action_idx < len(action_mask):
                    action_mask[action_idx] = True
                    move_to_index_map[action_idx] = i
            
            if not np.any(action_mask):
                print("PPOAgent error: No valid actions in mask")
                return random.choice(legal_moves)  # Return Move object
            
            # Get action from model
            action, _ = self.model.predict(observation, action_masks=action_mask, deterministic=self.deterministic)
            
            # Handle action format
            if hasattr(action, 'item'):
                chosen_action_idx = int(action.item())
            elif np.isscalar(action):
                chosen_action_idx = int(action)
            elif hasattr(action, '__len__') and len(action) > 0:
                chosen_action_idx = int(action[0])
            else:
                print(f"PPOAgent error: Unexpected action format: {type(action)}, {action}")
                return random.choice(legal_moves)  # Return Move object
            
            # Convert back to legal_moves index, then return the actual Move
            if chosen_action_idx in move_to_index_map:
                legal_move_idx = move_to_index_map[chosen_action_idx]
                return legal_moves[legal_move_idx] # ret the Move object, not the index
            else:
                print(f"PPOAgent error: Could not map chosen action {chosen_action_idx} back to legal move")
                return random.choice(legal_moves) # ret Move object
                
        except Exception as e:
            print(f"PPOAgent error: {e}")
            return random.choice(legal_moves) # ret Move object
    
    # Fallback: Rand legal action
    def _fallback_action(self, legal_moves):
        if isinstance(legal_moves, list) and len(legal_moves) > 0:
            return random.choice(range(len(legal_moves))) # ret index into legal_moves
        return 0

    # Reset agent state (PPO models are stateless)
    def reset(self):
        pass


# Manages Opp Pool for Self-Play Training
class SelfPlayOpponentManager:
    def __init__(self, stage2_model_path, stage3_model_path, current_model, main_env, update_frequency=2000, pool_weights=None):
        self.stage2_model_path = stage2_model_path
        self.stage3_model_path = stage3_model_path
        self.current_model = current_model
        self.main_env = main_env
        self.update_frequency = update_frequency
        self.episodes_since_update = 0
        
        # Default weights: more recent models get higher probability
        self.pool_weights = pool_weights or [0.5, 0.3, 0.2] # [current, stage3, stage2]
        
        # Load historical models
        self.stage2_agent = None
        self.stage3_agent = None
        self.current_agent = None
        
        self._load_historical_models()
        self._update_current_model()
        
        # Pool statistics
        self.selections = {'current': 0, 'stage3': 0, 'stage2': 0}
    
    # Load Stage 2-3 Models as Opps
    def _load_historical_models(self):
        print("Loading historical models for opponent pool...")
        
        # Create temp env for model loading
        # temp_env = DoudizhuEnv(opponent_agent=RandomAgent(), verbose=False)
        
        try:
            stage2_model = MaskablePPO.load(self.stage2_model_path, env=self.main_env)
            self.stage2_agent = PPOAgent(stage2_model, env_reference=self.main_env, deterministic=True)
            print(f"Stage 2 model loaded")
        except Exception as e:
            print(f"Failed to load Stage 2 model: {e}")
            self.stage2_agent = GreedyAgent()  # Fallback
        
        try:
            stage3_model = MaskablePPO.load(self.stage3_model_path, env=self.main_env)
            self.stage3_agent = PPOAgent(stage3_model, env_reference=self.main_env, deterministic=True)
            print(f"Stage 3 model loaded")
        except Exception as e:
            print(f"Failed to load Stage 3 model: {e}")
            self.stage3_agent = GreedyAgent()  # Fallback
        
        # temp_env.close()
    
    # Update curr model agewnt
    def _update_current_model(self):
        self.current_agent = PPOAgent(self.current_model, env_reference=self.main_env, deterministic=False) # slightly stochastic
    
    # Select Opp from pool based on weights
    def select_opponent(self):
        choice = np.random.choice(['current', 'stage3', 'stage2'], p=self.pool_weights)
        self.selections[choice] += 1
        
        if choice == 'current':
            return self.current_agent
        elif choice == 'stage3':
            return self.stage3_agent
        else:
            return self.stage2_agent
    
    # Episode Finished
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
                for opponent, count in self.selections.items():
                    print(f"  {opponent}: {count}/{total} ({100*count/total:.1f}%)")
                
                # Reset statistics
                self.selections = {'current': 0, 'stage3': 0, 'stage2': 0}
    
    # Get curr stats
    def get_stats(self):
        return {
            'episodes_since_update': self.episodes_since_update,
            'update_frequency': self.update_frequency,
            'pool_weights': self.pool_weights,
            'selections': self.selections.copy()
        }


def main():
    print("Stage 4 Training: PPO Self-Play")
    print("=" * 60)
    
    # Training parameters
    TOTAL_EPISODES = 15000 # 15k eps
    EVALUATION_EPISODES = 150 # More eps for robust self-play eval
    OPPONENT_UPDATE_FREQ = 2000 # update opp pool every 2k eps
    TARGET_WIN_RATE = 0.55 # target vs mixed opponent pool
    SEED = 42
    
    random.seed(SEED)
    np.random.seed(SEED)
    
    # Create Env (start with placeholder opp)
    print("Setting up self-play environment...")
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
    
    # Start with Greedy opponent (will be replaced by self-play manager)
    initial_opponent = GreedyAgent()
    env = DoudizhuEnv(
        opponent_agent=initial_opponent,
        verbose=False,
        seed=SEED,
        reward_config=reward_config
    )
    
    # Load Stage 3 model WITH Env
    stage3_model_path = "training/checkpoints/stage3/final_model"
    print(f"Loading Stage 3 model from: {stage3_model_path}")
    
    try:
        # Load the pre-trained model with Env
        model = MaskablePPO.load(stage3_model_path, env=env)
        print(f"Successfully loaded Stage 3 model")
        update_tensorboard_logging(model, "stage4_selfplay")
    except Exception as e:
        print(f"Failed to load Stage 3 model: {e}")
        print("Make sure Stage 3 training completed successfully")
        return None
    
    # Get Stage 4 configuration (for display/logging purposes)
    config = get_stage_config(stage=4)
    print(f"\nStage 4 PPO Configuration:")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Entropy Coefficient: {config.ent_coef} (minimal exploration)")
    
    # Using Stage 3 hyperparameters (USEFUKE for stability)
    print(f"Using Stage 3 model hyperparameters")
    
    # Setup self-play opponent manager
    stage2_model_path = "training/checkpoints/stage2/final_model"
    opponent_manager = SelfPlayOpponentManager(
        stage2_model_path=stage2_model_path,
        stage3_model_path=stage3_model_path,
        current_model=model,
        main_env=env,
        update_frequency=OPPONENT_UPDATE_FREQ,
        pool_weights=[0.5, 0.3, 0.2] # 50% current, 30% stage3, 20% stage2
    )
    
    print(f"\nSelf-Play Opponent Pool:")
    print(f"  Current Model: 50% (updated every {OPPONENT_UPDATE_FREQ} episodes)")
    print(f"  Stage 3 Model: 30%")
    print(f"  Stage 2 Model: 20%")
    
    # Setup callback with Stage 4 checkpoint directory
    checkpoint_dir = "training/checkpoints/stage4"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Self-play callback that manages opponent switching
    class SelfPlayCallback(DoudizhuTrainingCallback):
        def __init__(self, opponent_manager, **kwargs):
            super().__init__(**kwargs)
            self.opponent_manager = opponent_manager
            self.episode_count = 0
        
        def _on_step(self) -> bool:
            # Check if episode just ended
            if hasattr(self.locals, 'dones') and self.locals['dones'] is not None:
                if any(self.locals['dones']):  # Episode ended
                    self.episode_count += 1
                    
                    # Notify opponent manager
                    self.opponent_manager.episode_finished()
                    
                    # Select new opponent for next episode
                    new_opponent = self.opponent_manager.select_opponent()
                    self.model.env.set_opponent(new_opponent)
            
            return super()._on_step()
        
        def _log_progress(self):
            super()._log_progress()
            if hasattr(self, 'episode_count'):
                stats = self.opponent_manager.get_stats()
                print(f"Episodes: {self.episode_count}, Next pool update in: {stats['update_frequency'] - stats['episodes_since_update']}")
    
    callback = SelfPlayCallback(
        opponent_manager=opponent_manager,
        eval_freq=1500, # eval every 1500 eps (less frequent for self-play)
        save_freq=3000, # save every 3000 eps
        checkpoint_dir=checkpoint_dir,
        log_freq=300, # log every 300 eps
        verbose=1
    )
    
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Tensorboard logs: logs/tensorboard/stage4_selfplay_*")
    
    # Calculate total timesteps
    estimated_steps_per_episode = 100
    total_timesteps = TOTAL_EPISODES * estimated_steps_per_episode
    
    print(f"\nStarting Stage 4 self-play training...")
    print(f"Target episodes: {TOTAL_EPISODES}")
    print(f"Estimated timesteps: {total_timesteps}")
    print(f"Opponent pool update frequency: {OPPONENT_UPDATE_FREQ} episodes")
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
    
    # Final comprehensive evaluation
    print(f"\nRunning final evaluation against all opponent types...")
    
    # Test against historical rule-based opponents
    print(f"Evaluating vs Random agent ({EVALUATION_EPISODES} episodes)...")
    random_opponent = RandomAgent()
    # eval_env.set_opponent(opponent_agent=random_opponent, verbose=False, seed=SEED, reward_config=reward_config)
    eval_env = DoudizhuEnv(opponent_agent=random_opponent, verbose=False, seed=SEED, reward_config=reward_config)
    random_results = evaluate_agent(model, eval_env, num_episodes=EVALUATION_EPISODES)

    print(f"Evaluating vs Conservative agent ({EVALUATION_EPISODES} episodes)...")
    conservative_opponent = ConservativeAgent()
    eval_env.set_opponent(conservative_opponent)
    conservative_results = evaluate_agent(model, eval_env, num_episodes=EVALUATION_EPISODES)
    
    print(f"Evaluating vs Greedy agent ({EVALUATION_EPISODES} episodes)...")
    greedy_opponent = GreedyAgent()
    eval_env = DoudizhuEnv(greedy_opponent)
    # eval_env.set_opponent(greedy_opponent)
    greedy_results = evaluate_agent(model, eval_env, num_episodes=EVALUATION_EPISODES)

    # DEBUGGING PRINTS 
    print(f"About to evaluate - Current player: {eval_env.game.state.curr_player}")
    print(f"RL player is: {eval_env.rl_player}")
    print(f"Opponent player is: {eval_env.opponent_player}")

    # Test against historical self versions
    print(f"Evaluating vs Stage 2 model ({EVALUATION_EPISODES} episodes)...")
    eval_env.set_opponent(opponent_manager.stage2_agent)
    stage2_results = evaluate_agent(model, eval_env, num_episodes=EVALUATION_EPISODES)

    print(f"Evaluating vs Stage 3 model ({EVALUATION_EPISODES} episodes)...")
    eval_env.set_opponent(opponent_manager.stage3_agent)
    stage3_results = evaluate_agent(model, eval_env, num_episodes=EVALUATION_EPISODES)
    

    print(f"\nSTAGE 4 FINAL RESULTS")
    print("=" * 60)
    print(f"VS RULE-BASED OPPONENTS:")
    print(f"  Greedy:      {greedy_results['win_rate']:.3f} (avg len: {greedy_results['avg_episode_length']:.1f})")
    print(f"  Conservative: {conservative_results['win_rate']:.3f} (avg len: {conservative_results['avg_episode_length']:.1f})")
    print(f"  Random:      {random_results['win_rate']:.3f} (avg len: {random_results['avg_episode_length']:.1f})")
    
    print(f"\nVS HISTORICAL SELF-VERSIONS:")
    print(f"  Stage 3 Model: {stage3_results['win_rate']:.3f} (avg len: {stage3_results['avg_episode_length']:.1f})")
    print(f"  Stage 2 Model: {stage2_results['win_rate']:.3f} (avg len: {stage2_results['avg_episode_length']:.1f})")
    
    print(f"\nOVERALL PERFORMANCE:")
    print(f"  Invalid Actions: {greedy_results['invalid_actions']}") # shouldn't have any...
    print(f"  Cards on Loss: {greedy_results['avg_cards_on_loss']:.1f}")
    
    # Role-specific performance
    print(f"\nRole-Specific Performance (vs Greedy):")
    for role, stats in greedy_results['role_performance'].items():
        print(f"  {role.capitalize()}: {stats['win_rate']:.3f} ({stats['games']} games)")
    
    # Save final model and comprehensive results
    final_model_path = os.path.join(checkpoint_dir, "final_model")
    model.save(final_model_path)
    
    # Comprehensive results summary
    results_summary = {
        'stage': 4,
        'training_type': 'self_play',
        'training_episodes': TOTAL_EPISODES,
        'training_time': training_time,
        'opponent_pool_stats': opponent_manager.get_stats(),
        'vs_greedy': greedy_results,
        'vs_conservative': conservative_results,
        'vs_random': random_results,
        'vs_stage3_model': stage3_results,
        'vs_stage2_model': stage2_results
    }
    
    final_results_path = os.path.join(checkpoint_dir, "final_results.json")
    with open(final_results_path, 'w') as f:
        json.dump(convert_for_json(results_summary), f, indent=2)
    
    print(f"\nFinal model saved: {final_model_path}")
    print(f"Final results saved: {final_results_path}")
    
    # Complete performance evolution
    print(f"\nPerformance:")
    print(f"Stage 4: Random {random_results['win_rate']:.0%} | Conservative {conservative_results['win_rate']:.0%} | Greedy {greedy_results['win_rate']:.0%}")
    print(f"         vs Stage3: {stage3_results['win_rate']:.0%} | vs Stage2: {stage2_results['win_rate']:.0%}")
    
    # Training insights
    print(f"\nSELF-PLAY INSIGHTS:")
    total_selections = sum(opponent_manager.selections.values())
    if total_selections > 0:
        print(f"Final opponent distribution:")
        for opponent, count in opponent_manager.selections.items():
            print(f"  {opponent}: {100*count/total_selections:.1f}%")
    
    # Success evaluation
    avg_performance = np.mean([
        greedy_results['win_rate'], 
        conservative_results['win_rate'],
        stage3_results['win_rate']
    ])
    
    print(f"\nAverage performance vs challenging opponents: {avg_performance:.1%}")
    
    if avg_performance >= TARGET_WIN_RATE:
        print(f"SUCCESS! Self-play training achieved strong performance!")
    else:
        print(f"Performance at {avg_performance:.1%} - consider extended self-play training")
    
    # Cleanup
    env.close()
    eval_env.close()
    
    print(f"\nStage 4 self-play training complete!")
    print(f"Total training time: {training_time/3600:.1f} hours")
    
    return results_summary


if __name__ == "__main__":
    # Pre-training checks
    try:
        print("Running pre-training compatibility check...")
        
        # Check if required models exist
        stage2_path = "training/checkpoints/stage2/final_model"
        stage3_path = "training/checkpoints/stage3/final_model"
        
        missing_models = []
        if not os.path.exists(stage2_path + ".zip") and not os.path.exists(stage2_path):
            missing_models.append("Stage 2")
        if not os.path.exists(stage3_path + ".zip") and not os.path.exists(stage3_path):
            missing_models.append("Stage 3")
        
        if missing_models:
            raise FileNotFoundError(f"Missing models: {', '.join(missing_models)}. Complete previous stages first.")
        
        # Test PPOAgent wrapper
        test_env = DoudizhuEnv(opponent_agent=RandomAgent(), verbose=False)
        test_model = MaskablePPO.load(stage3_path, env=test_env)
        test_agent = PPOAgent(test_model, env_reference=test_env) # add env_reference

        obs, info = test_env.reset()
        legal_moves = test_env.get_legal_moves() # get legal moves, not action mask
        game_state = test_env.game.state # get game state

        action = test_agent.choose_action(legal_moves, game_state) # correct params

        if not hasattr(action, 'combo_type'): # check if it's a Move object
            raise ValueError(f"PPOAgent returned invalid action type: {type(action)}")
        
        test_env.close()
        
        print("Self-play compatibility check passed")
        print(f"  PPOAgent wrapper working")
        print(f"  Stage 2 and Stage 3 models found")
        
    except Exception as e:
        print(f"Pre-training check failed: {e}")
        exit(1)
    
    results = main()

