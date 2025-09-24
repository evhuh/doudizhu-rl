# doudizhu/doudizhu_env.py

# Lighweight Gym Wrapper around DoudizhuGame
# - RL agent always P1
# Space:
#   - Work with a variable action space for easy display and use (game.get_legal_moves())
#   - Fixed action space for RL training (PPO, DQN)
# - Ret Observation Vector [P1 counts(15) opp hand size, last_move_type, last_move_rank, role_indicator, turn_indicator]
# - Small Intermediate rewards
# - Large Terminal rewards
# + Rand opp by default (placeholder for other agents



import sys
import os
import numpy as np
import random
from typing import Dict, List, Tuple, Any, Optional
import gymnasium as gym
from gymnasium import spaces

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.insert(0, project_root)

from doudizhu.utils import (NUM_RANKS, RANK_VALUES, RANKS, COMBO_SINGLE, COMBO_PAIR, COMBO_TRIPLE, COMBO_STRAIGHT, COMBO_BOMB, COMBO_ROCKET, COMBO_PASS)
from doudizhu.game_engine import DoudizhuGame, Move, GameState, get_legal_moves
from doudizhu.action_space import ActionSpace

from agents.random_agent import RandomAgent
from agents.greedy_agent import GreedyAgent
from agents.conservative_agent import ConservativeAgent



# Gym-like env for 1v1
# Single-agent RL Training against another agent
class DoudizhuEnv(gym.Env):
    # Sets up Env
    def __init__(self, opponent_agent=None, verbose=False, seed=None, reward_config=None):
        super(DoudizhuEnv, self).__init__()

        self.verbose = verbose
        self.rng = random.Random(seed)

        # Opp Agent
        if opponent_agent is None:
            self.opponent_agent = RandomAgent(seed=seed)
            if self.verbose:
                print(f"Created default RandomAgent opponent")
        else:
            self.opponent_agent = opponent_agent

        # Reward configurtion (conservative_shaping by default)
        self.reward_config = reward_config or {
            'win_reward': 5.0,
            'loss_penalty': -5.0,
            'step_penalty': -0.001,
            'hand_reduction_bonus': 0.02,
            'invalid_action_penalty': -0.5, # increased from -0.1
            'endgame_urgency_bonus': 0.1,
            'role_bonus_landlord': 0.0,
            'role_bonus_farmer': 0.0
        }

        self.action_space_handler = ActionSpace()
        self.num_actions = len(self.action_space_handler.actions)
        if self.verbose:
            print(f"Init'ed fixed action space with {self.num_actions} actions")
        
        # Defined gymnasium spaces
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(28,), 
            dtype=np.float32
        )

        self.game = DoudizhuGame(rng=self.rng)
        self.rl_player = 1 # RL agent is always P1
        self.opponent_player = 2 # Opponent is always P2
        
        self.episode_step = 0
        self.max_episode_steps = 200 # prevent inf games...prob don't need allat

        # Tracking Metrics
        self.episode_history = []
        self.episode_stats = {}
        self.last_hand_size = 17
        self.consecutive_passes = 0
        self.cards_played_this_episode = 0

        self.observation_dim = 28


    # ACTION MASKING METHODS ================================================================

    # Ret Bool Mask of Valid Actions
    # SB3 MaskablePPO
    def action_masks(self) -> np.ndarray:
        if self.game.is_game_over(): 
            return np.zeros(self.num_actions, dtype=bool)
        if self.game.state.curr_player != self.rl_player:
            return np.zeros(self.num_actions, dtype=bool)

        # 1. Get dynamic legal moves from game engine
        legal_moves = self.game.get_legal_moves()
        mask = np.zeros(self.num_actions, dtype=bool)

        # 2. Mark corresp fixed actions as valid
        for move in legal_moves:
            action_idx = self.move_to_action(move)
            if action_idx is not None:
                mask[action_idx] = True
            elif self.verbose:
                print(f"Warning: Count not map move ot action: {move}")
        
        return mask

    # Convert Move Obj to Fixed Action Index
    def move_to_action(self, move: Move) -> Optional[int]:
        return self.action_space_handler.move_to_action(move)

    # Convert Action Index to Move Obj
    def action_to_move(self, action_idx: int) -> Move:
        if action_idx >= self.num_actions:
            raise ValueError(f"Action index {action_idx} out of range [0, {self.num_actions-1}]")
        return self.action_space_handler.actions[action_idx]


    # Get Actual Legal Move Objects
    def get_legal_moves(self) -> List[Move]:
        return self.game.get_legal_moves()
    
    # Get Indices of Legal Actions for Curr State; just ret indices 0 to len(legal_moves)-1
    def legal_actions(self) -> List[int]:
        if self.game.is_game_over():
            return []
        if self.game.state.curr_player != self.rl_player:
            return [] # not RL agent's turn

        # legal_moves = self.game.get_legal_moves()
        # return list(range(len(legal_moves)))

        mask = self.action_masks()
        return [i for i, valid in enumerate(mask) if valid]
    

    # CURRICULUM LEARNING SUPPORT ================================================================
    # Change Opp Agent
    def set_opponent(self, opponent_agent):
        self.opponent_agent = opponent_agent
        if self.verbose:
            print(f"Opp changed to: {type(opponent_agent).__name__}")
    
    # Get Current Opp Type for Logging
    def get_opponent_strength(self) -> str:
        return type(self.opponent_agent).__name__


    # ENV METHODS ================================================================

    # Reset env for new epsiode
    # RL agent always P1, oppoent always P2 (role can be ll, farmer, or rand)
    def reset(self, rl_agent_role: str='random', seed=None, options=None) -> np.ndarray:
        if seed is not None: self.seed(seed)

        # Start new game with RL agent as P1
        if rl_agent_role.lower() == 'random':
            self.game.start_new_game(p1_role='random', p2_role='random')
        elif rl_agent_role.lower() == 'landlord':
            self.game.start_new_game(p1_role='landlord', p2_role='farmer') # RL agent wants landlord, opp gets farmer
        elif rl_agent_role.lower() == 'farmer':
            self.game.start_new_game(p1_role='farmer', p2_role='landlord') # RL agent wants farmer, opp gets landlord
        else:
            raise ValueError("rl_agent_role must be 'landlord', 'farmer', or 'random'")

        # Reset Tracking
        self.episode_step = 0
        self.episode_history = []
        self.last_hand_size = 17
        self.consecutive_passes = 0
        self.cards_played_this_episode = 0

        rl_role = self.game.get_player_role(self.rl_player)
        if self.verbose:
            print(f"Environment reset - RL Agent is Player {self.rl_player} ({rl_role})")

        # If opponent goes 1st (they're landlord), let them play
        if self.game.state.curr_player == self.opponent_player:
            self._opponent_turn()
        
        observation = self._get_observation()
        info = {
            "episode_step": self.episode_step,
            "rl_agent_role": rl_role,
            "current_player": self.game.state.curr_player,
            "action_mask": self.action_masks(),
            "opponent_type": self.get_opponent_strength()
        }
        return observation, info
    
    
    # RL Agent Turn
    # Take Single Step in Env
    # Reward Signal: Win = +1, Loss = -1, w/ intermediate rewards
    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # CHECK: Validate Action Index is in Range
        if action_idx >= self.num_actions:
            raise ValueError(f"Action Index {action_idx} out of range [0, {self.num_actions-1}]")
        
        # CHECK: Game State Validation
        if not self.game.validate_state():
            raise RuntimeError("Game state corruption detected")
        if self.game.is_game_over():
            raise ValueError("Game is already over. Call reset() to start new episode.")
        if self.game.state.curr_player != self.rl_player:
            raise ValueError("Not RL agent's turn")
        
        # CHECK: If Action is Valid Using Mask
        valid_actions = self.action_masks()
        if not valid_actions[action_idx]:
            # Invalid action handling
            if self.verbose:
                attempted_move = self.action_to_move(action_idx)
                print(f"Invalid action attempted: {action_idx} ({attempted_move})")
            
            # Ret penality and continue episode
            observation = self._get_observation()
            info = {
                "invalid_action": True, 
                "episode_step": self.episode_step,
                "action_mask": self.action_masks(),
                "cards_played_this_episode": self.cards_played_this_episode
            }
            return observation, self.reward_config['invalid_action_penalty'], False, False, info

        
        # Apply RL agent's move
        move = self.action_to_move(action_idx) # convert action to move and apply
        prev_hand_size = sum(self.game.state.player1_vec) # hand size before play
        
        success = self.game.play_move(move)
        if not success: # NOTE: Does not happen w masking
            if self.verbose:
                print(f"Warning: Failed to apply move {move}")
            info = {"failed_move": True, "episode_step": self.episode_step, "action_mask": self.action_masks()}
            return observation, -0.5, False, False, info
        
        self.episode_step += 1
        self.episode_history.append((self.rl_player, move))

        # CHECK
        if not self.game.validate_state():
            raise RuntimeError("Game state corruption after move")

        # Track Move Stats
        new_hand_size = sum(self.game.state.player1_vec)
        cards_played = prev_hand_size - new_hand_size
        self.cards_played_this_episode += cards_played


        if move.combo_type == COMBO_PASS:
            self.consecutive_passes += 1
        else:
            self.consecutive_passes = 0


        # Intermediate Reward Calc
        reward = self._calc_int_reward(move, prev_hand_size, new_hand_size)

        # CHECK: If Game Over After RL Agent's Move
        if self.game.is_game_over():
            return self._handle_game_end(reward)
        # CHECK: Episode Len Limit
        if self.episode_step >= self.max_episode_steps:
            if self.verbose:
                print("Episode terminated: Max steps reached")
            info = {
                "timeout": True, 
                "episode_step": self.episode_step,
                "action_mask": self.action_masks(),
                "cards_remaining": new_hand_size
            }
            return self._get_observation(), 0, False, True, info


        # Let opponent play their turn
        self._opponent_turn()

        # CHECK
        # Check if game over after opponent's move
        if self.game.is_game_over():
            return self._handle_game_end(reward)

        # Game Continues
        observation = self._get_observation()
        terminated = False
        truncated = False
        info = {
            "episode_step": self.episode_step,
            "action_mask": self.action_masks(),
            "cards_played": cards_played,
            "cards_remaining": new_hand_size,
            "consecutive_passes": self.consecutive_passes,
            "cards_played_this_episode": self.cards_played_this_episode
        }

        return observation, reward, terminated, truncated, info
    

    # Intermeditate Reward Calcs
    def _calc_int_reward(self, move: Move, prev_hand_size: int, new_hand_size: int) -> float:
        reward = self.reward_config['step_penalty']

        if move.combo_type == COMBO_PASS:
            # Small penalty for consecutive passes
            if self.consecutive_passes > 2:
                reward -= 0.05
            return reward
        
        cards_played = prev_hand_size - new_hand_size
        opponent_hand_size = sum(self.game.state.player2_vec)
        total_cards_remaining = new_hand_size + opponent_hand_size

        # Base reward for playing cards
        reward += self.reward_config['hand_reduction_bonus'] * cards_played

        # Endgame urgency - play more aggressively when opponent is close to winning
        if opponent_hand_size <= 5:
            reward += self.reward_config['endgame_urgency_bonus'] * cards_played

        # Role-based bonuses
        rl_role = self.game.get_player_role(self.rl_player)
        if rl_role == 'landlord':
            reward += self.reward_config['role_bonus_landlord']
        else:
            reward += self.reward_config['role_bonus_farmer']

        # Efficiency bonus for multi-card plays early/mid game
        if total_cards_remaining > 15 and cards_played >= 3:
            reward += 0.05

        # Strategic combo bonuses
        if move.combo_type == COMBO_STRAIGHT and cards_played >= 5:
            reward += 0.1  # Good straight play
        elif move.combo_type in [COMBO_BOMB, COMBO_ROCKET]:
            # Bonus for bombs/rockets in endgame
            if total_cards_remaining <= 10:
                reward += 0.2

        return reward
    

    # PRIVATE METHODS ================================================================
    
    # Describe Current State to RL Agent
    # Format: (28 dims):
    # [0-14] : RL agent hand counts (15 dims, scaled to [0,1])
    # [15] : Opp hand size (1 dim, scaled to [0,1])
    # [16-23] : Last move type one-hot (8 dims: [NONE, PASS, SINGLE, PAIR, TRIPLE, STRAIHGT, BOMB, ROCKET])
    # [24-25] : Role one-hot (2 dims: [ll, farmer])
    # [26-27] : Turn one-hot (2 dims: [RL turn, Opp turn])

    def _get_observation(self) -> np.ndarray:
        # RL agent's hand normalized by 4 (max 4 cards per rank), jokers normalized by 2
        hand_counts = self.game.state.player1_vec
        rl_hand = [
            hand_counts[i] / 4.0 if i < 13 else hand_counts[i] / 2.0
            for i in range(NUM_RANKS)
        ]

        # Opp's hand size normalized by 17 (max hand size)
        opponent_hand_size = [sum(self.game.state.player2_vec) / 17.0]
        
        # Last move type (one-hot: [PASS, single, pair, triple, straight, bomb, rocket])
        last_move_type = [0.0] * 7
        last_move_rank_norm = [0.0]

        if self.game.state.last_move is None or self.game.state.last_move.combo_type == COMBO_PASS:
            last_move_type[0] = 1.0 # None / PASS
        else: # self.game.state.last_move and self.game.state.last_move.combo_type != COMBO_PASS:
            # map combo types to indices
            move_type_map = {
                COMBO_SINGLE: 1,
                COMBO_PAIR: 2,
                COMBO_TRIPLE: 3,
                COMBO_STRAIGHT: 4,
                COMBO_BOMB: 5,
                COMBO_ROCKET: 6
            }
            if self.game.state.last_move.combo_type in move_type_map:
                idx = move_type_map[self.game.state.last_move.combo_type]
                last_move_type[idx] = 1.0

            # Last move rank scaled to [0, 1], -1 becomes 0.0 (no rank)
            # handle rank encoding (soem combos dont have meaningful primary_rank)
            if self.game.state.last_move.primary_rank:
                rank_value = RANK_VALUES[self.game.state.last_move.primary_rank]
                last_move_rank_norm = [rank_value / (NUM_RANKS - 1)]

        # Role Indicator, one-hot: [landlord, farmer] # Turn Indicator, one-hot: [rl_turn, opp_turn]
        rl_role = self.game.get_player_role(self.rl_player)
        role_onehot = [1.0, 0.0] if rl_role == 'landlord' else [0.0, 1.0] 
        turn_onehot = [1.0, 0.0] if self.game.state.curr_player == self.rl_player else [0.0, 1.0]

        # Create Obs
        observation = rl_hand + opponent_hand_size + last_move_type + last_move_rank_norm + role_onehot + turn_onehot
        return np.array(observation, dtype=np.float32)
    

    # Opp Agent Turn
    def _opponent_turn(self):
        while (not self.game.is_game_over() and self.game.state.curr_player == self.opponent_player):
            legal_moves = self.game.get_legal_moves()
            if not legal_moves:
                if self.verbose: print("Warning: No legal moves for opponent")
                break
            
            # Get opp move
            move = self.opponent_agent.choose_action(legal_moves, self.game.state)
            success = self.game.play_move(move)
            if not success:
                if self.verbose:
                    print(f"Warning: Opp made invalid move: {move}")
                # handle invalid move (fallback to pass or random legal move)
                pass_moves = [m for m in legal_moves if m.combo_type == COMBO_PASS]
                if pass_moves: self.game.play_move(pass_moves[0])
                break
        
            self.episode_history.append((self.opponent_player, move))

            # Handle consecuitve passes or game state changes
            if (self.game.state.last_move is None or self.game.state.curr_player == self.rl_player):
                break


    # Handle Game Termination and Ret Final Step Info
    def _handle_game_end(self, intermediate_reward: float = 0) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        observation = self._get_observation()
        
        result = self.game.get_game_result()
        final_reward = result[f'player{self.rl_player}_reward']
        
        # Apply reward config scaling
        if final_reward > 0: # win
            final_reward = self.reward_config['win_reward']
        else: # loss
            final_reward = self.reward_config['loss_penalty']
        
        total_reward = final_reward + intermediate_reward
        
        # Logging Info
        info = {
            "episode_step": self.episode_step,
            "winner": result['winner'],
            "rl_agent_won": result['winner'] == self.rl_player,
            "game_length": len(self.episode_history),
            "rl_agent_role": self.game.get_player_role(self.rl_player),
            "final_reward": final_reward,
            "intermediate_reward": intermediate_reward,
            "total_reward": total_reward,
            "cards_remaining": sum(self.game.state.player1_vec),
            "opponent_cards_remaining": sum(self.game.state.player2_vec),
            "cards_played_this_episode": self.cards_played_this_episode,
            "opponent_type": self.get_opponent_strength(),
            "action_mask": np.zeros(self.num_actions, dtype=bool) # game over
        }
        
        if self.verbose:
            win_status = "WON" if info["rl_agent_won"] else "LOST"
            print(f"Game ended! RL Agent {win_status} - Total reward: {total_reward:.3f}")
        
        return observation, total_reward, True, False, info
    

    def render(self, mode='human'):
        if mode == 'human':
            self.game.print_game_state()
            if self.verbose:
                mask = self.action_masks()
                valid_actions = [i for i, valid in enumerate(mask) if valid]
                print(f"Valid action idnices: {valid_actions}")
                print(f"Valid moves: {[str(self.action_to_move(i)) for i in valid_actions]}")
        else:
            raise NotImplementedError(f"Render mode '{mode}' not supported")
    
    def seed(self, seed: int):
        # random.seed(seed)
        # np.random.seed(seed)
        self.rng.seed(seed)
    
    # Clean Up Resources
    # TODO: Use for PyGame GUI when rendering game...?
    def close(self):
        pass



# TESTING ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# def test():
#     # Initialize env
#     env = DoudizhuEnv(verbose=True, seed=42)
    
#     # Test 1: Basic setup and spaces
#     print(f"\nENVIRONMENT SETUP")
#     print(f"Action space: {env.action_space}")
#     print(f"Observation space: {env.observation_space}")
#     print(f"Total actions in fixed space: {env.num_actions}")
    
#     # Test 2: Reset and observation structure
#     print(f"\nRESET TEST")
#     obs, info = env.reset(rl_agent_role='landlord')
#     print(f"Initial observation shape: {obs.shape}")
#     print(f"Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
#     print(f"Reset info: {info}")
    
#     # Verify observation structure (28 dims)
#     print(f"\nObservation breakdown:")
#     print(f"  RL hand counts [0-14]: {obs[:15]}")
#     print(f"  Opponent hand size [15]: {obs[15]:.3f}")
#     print(f"  Last move type [16-22]: {obs[16:23]}")
#     print(f"  Last move rank [23]: {obs[23]:.3f}")
#     print(f"  Role [24-25]: {obs[24:26]} ({'landlord' if obs[24] > obs[25] else 'farmer'})")
#     print(f"  Turn [26-27]: {obs[26:28]} ({'RL' if obs[26] > obs[27] else 'opponent'})")
    
#     assert len(obs) == 28, f"Expected 28 dims, got {len(obs)}"
#     assert np.all((obs >= 0) & (obs <= 1)), "All observation values should be in [0,1]"
    
#     # Test 3: Action masking functionality
#     print(f"\nACTION MASKING TESTS")
#     mask = env.action_masks()
#     legal_actions = env.legal_actions()
#     legal_moves = env.get_legal_moves()
    
#     print(f"Action mask shape: {mask.shape}")
#     print(f"Valid actions count: {np.sum(mask)}")
#     print(f"Legal actions from env: {len(legal_actions)}")
#     print(f"Legal moves from game: {len(legal_moves)}")
    
#     # Verify mask consistency
#     masked_indices = [i for i, valid in enumerate(mask) if valid]
#     assert set(legal_actions) == set(masked_indices), "Mask and legal_actions() mismatch!"
    
#     print(f"Sample legal moves: {[str(move) for move in legal_moves[:3]]}")
    
#     # Test 4: Valid action execution
#     print(f"\nVALID ACTION TEST")
#     if legal_actions:
#         # Try to find a non-pass action
#         chosen_action = legal_actions[0]
#         chosen_move = env.action_to_move(chosen_action)
        
#         for action_idx in legal_actions:
#             move = env.action_to_move(action_idx)
#             if move.combo_type != COMBO_PASS:
#                 chosen_action = action_idx
#                 chosen_move = move
#                 break
        
#         print(f"Executing action {chosen_action}: {chosen_move}")
        
#         obs_before = env._get_observation()
#         hand_size_before = sum(env.game.state.player1_vec)
        
#         obs, reward, terminated, truncated, info = env.step(chosen_action)
        
#         hand_size_after = sum(env.game.state.player1_vec)
#         cards_played = hand_size_before - hand_size_after
        
#         print(f"Result: reward={reward:.3f}, terminated={terminated}, truncated={truncated}")
#         print(f"Cards played: {cards_played}, hand size: {hand_size_before} -> {hand_size_after}")
#         print(f"Step info keys: {list(info.keys())}")
    
#     # Test 5: Invalid action handling
#     print(f"\nINVALID ACTION TEST")
#     # Reset to ensure we're in RL agent's turn
#     obs, info = env.reset(rl_agent_role='random')
    
#     # Find an invalid action (ie. not in mask)
#     mask = env.action_masks()
#     invalid_actions = [i for i, valid in enumerate(mask) if not valid]
    
#     if invalid_actions:
#         invalid_action = invalid_actions[0]
#         print(f"Testing invalid action: {invalid_action}")
        
#         obs, reward, terminated, truncated, info = env.step(invalid_action)
        
#         print(f"Invalid action result: reward={reward:.3f}, terminated={terminated}")
#         print(f"Info contains 'invalid_action': {info.get('invalid_action', False)}")
        
#         assert reward == env.reward_config['invalid_action_penalty'], "Wrong invalid action penalty"
#         assert info.get('invalid_action', False), "Should mark as invalid action"
#         assert not terminated, "Invalid action shouldn't terminate episode"
    
#     # Test 6: Full episode simulation
#     print(f"\nEP SIMULATION")
#     obs, info = env.reset(rl_agent_role='landlord')
    
#     step_count = 0
#     total_reward = 0
#     max_steps = 15
    
#     print(f"Starting episode with RL agent as {info['rl_agent_role']}")
    
#     while not env.game.is_game_over() and step_count < max_steps:
#         if env.game.state.curr_player == env.rl_player:
#             legal_actions = env.legal_actions()
            
#             if not legal_actions:
#                 print(f"No legal actions at step {step_count}")
#                 break
            
#             # Choose action (prefer non-pass)
#             action = legal_actions[0]
#             for act in legal_actions:
#                 move = env.action_to_move(act)
#                 if move.combo_type != COMBO_PASS:
#                     action = act
#                     break
            
#             move_str = str(env.action_to_move(action))
#             print(f"Step {step_count + 1}: Action {action} ({move_str})")
            
#             obs, reward, terminated, truncated, info = env.step(action)
#             total_reward += reward
            
#             print(f"  -> Reward: {reward:.3f}, Cards remaining: {info.get('cards_remaining', 'N/A')}")
            
#             if terminated:
#                 print(f"Episode ended: {info}")
#                 break
        
#         step_count += 1
    
#     print(f"Episode completed after {step_count} steps")
#     print(f"Total reward: {total_reward:.3f}")
    
#     # Test 7: Reward configuration
#     print(f"\nREWARD CONFIG TEST")
#     print("Current reward config:")
#     for key, value in env.reward_config.items():
#         print(f"  {key}: {value}")
    
#     # Test 8: Opponent switching
#     print(f"\nOPP SWITCHING TEST")
#     from agents.greedy_agent import GreedyAgent
#     from agents.conservative_agent import ConservativeAgent
    
#     original_opp = env.get_opponent_strength()
#     print(f"Original opponent: {original_opp}")
    
#     # Switch to greedy agent
#     env.set_opponent(GreedyAgent())
#     new_opp = env.get_opponent_strength()
#     print(f"New opponent: {new_opp}")
    
#     # Switch to conservative agent  
#     env.set_opponent(ConservativeAgent())
#     final_opp = env.get_opponent_strength()
#     print(f"Final opponent: {final_opp}")
    
#     env.close()
#     print(f"\nAll tests passed!!!")

# if __name__ == "__main__":
#     test()

