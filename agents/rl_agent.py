# agents/rl_agent.py

import sys
import os
import random
import numpy as np
from typing import List, Optional

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.insert(0, project_root)

from base_agent import Agent
from doudizhu.game_engine import Move, GameState
from doudizhu.doudizhu_env import DoudizhuEnv
from sb3_contrib import MaskablePPO


class RLAgent(Agent):
    def __init__(self, model_path="training/checkpoints/stage5/final_model", deterministic=True):
        super().__init__("RL Agent")
        self.model_path = model_path
        self.deterministic = deterministic
        self.model = None
        self.temp_env = None
        
        # Try to load the model
        self._load_model()
    
    def _load_model(self):
        """Load the trained PPO model"""
        try:
            # Create a temporary environment for model loading
            # We need this for the observation space and action space
            from agents.random_agent import RandomAgent
            self.temp_env = DoudizhuEnv(opponent_agent=RandomAgent(), verbose=False)
            
            # Load the model
            self.model = MaskablePPO.load(self.model_path, env=self.temp_env)
            print(f"Successfully loaded RL model from: {self.model_path}")
            
        except FileNotFoundError:
            print(f"ㄨ ERROR: RL model not found at {self.model_path}")
            print("   Please ensure Stage 5 training has completed successfully.")
            raise FileNotFoundError(f"RL model not found: {self.model_path}")
        except Exception as e:
            print(f"ㄨ ERROR: Failed to load RL model: {e}")
            print("   Please check that the model file is valid and Stage 5 completed.")
            raise RuntimeError(f"Failed to load RL model: {e}")
    
    def choose_action(self, legal_moves: List[Move], game_state: Optional[GameState] = None) -> Move:
        """Choose action using the trained PPO model"""
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        # Fallback if model failed to load
        if self.model is None or self.temp_env is None:
            print("!! RL model not available, using random fallback")
            return random.choice(legal_moves)
        
        try:
            # Set the game state in our temporary environment
            if game_state is not None:
                self.temp_env.game.state = game_state
            
            # Get observation from the environment
            observation = self.temp_env._get_observation()
            
            if observation.shape != (28,):
                print(f"!! Invalid observation shape {observation.shape}, using random fallback")
                return random.choice(legal_moves)
            
            # Create action mask for valid moves
            action_mask = np.zeros(self.model.action_space.n, dtype=bool)
            move_to_index_map = {}
            
            for i, move in enumerate(legal_moves):
                action_idx = self.temp_env.move_to_action(move)
                if action_idx is not None and 0 <= action_idx < len(action_mask):
                    action_mask[action_idx] = True
                    move_to_index_map[action_idx] = i
            
            if not np.any(action_mask):
                print("!! No valid actions in mask, using random fallback")
                return random.choice(legal_moves)
            
            # Get action from model
            action, _ = self.model.predict(
                observation, 
                action_masks=action_mask, 
                deterministic=self.deterministic
            )
            
            # Convert action to proper format
            if hasattr(action, 'item'):
                chosen_action_idx = int(action.item())
            elif np.isscalar(action):
                chosen_action_idx = int(action)
            elif hasattr(action, '__len__') and len(action) > 0:
                chosen_action_idx = int(action[0])
            else:
                print(f"!! Unexpected action format: {type(action)}, using random fallback")
                return random.choice(legal_moves)
            
            # Map back to legal move
            if chosen_action_idx in move_to_index_map:
                legal_move_idx = move_to_index_map[chosen_action_idx]
                chosen_move = legal_moves[legal_move_idx]
                return chosen_move
            else:
                print(f"!! Could not map action {chosen_action_idx} to legal move, using random fallback")
                return random.choice(legal_moves)
                
        except Exception as e:
            print(f"!! RL Agent error: {e}, using random fallback")
            return random.choice(legal_moves)
    
    def game_started(self, role: str, player_number: int):
        """Called when a new game starts"""
        super().game_started(role, player_number)
        # Reset any internal state if needed
        pass
    
    def game_ended(self, won: bool):
        """Called when game ends"""
        super().game_ended(won)
        # Could log performance here if desired
        pass
    
    def __del__(self):
        """Cleanup when agent is destroyed"""
        if hasattr(self, 'temp_env') and self.temp_env is not None:
            try:
                self.temp_env.close()
            except:
                pass
        