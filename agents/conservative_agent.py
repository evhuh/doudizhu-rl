# agents/conservative_agent.py

# Strat:
# 1. Avoid playing bombs/rockets unless necessary
# 2. Prefer singles over multi-card combos to preserve flexibility
# 3. Save high-value cards (A, 2, jokers, JOKERS) for crucial moments
# 4. Pass more often to avoid revealing hand strength
# 5. Play more aggressively only when opp has few cards



import sys
import os
from typing import List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_agent import Agent
from doudizhu.utils import (COMBO_PASS, COMBO_SINGLE, COMBO_PAIR, COMBO_TRIPLE, COMBO_STRAIGHT, COMBO_BOMB, COMBO_ROCKET, RANK_VALUES)
from doudizhu.game_engine import Move, GameState



class ConservativeAgent(Agent):
    def __init__(self):
        super().__init__("Conservative Agent")
        self.high_value_ranks = ['A', '2', 'joker', 'JOKER']
    
    def choose_action(self, legal_moves: List[Move], game_state: Optional[GameState] = None) -> Move:
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        # Separate moves by type
        pass_moves = [m for m in legal_moves if m.combo_type == COMBO_PASS]
        play_moves = [m for m in legal_moves if m.combo_type != COMBO_PASS]
        
        # If only pass available, pass
        if not play_moves:
            return pass_moves[0]
        
        # Get game context for decision making
        opponent_hand_size = self._get_opponent_hand_size(game_state)
        last_move = game_state.last_move if game_state else None
        
        # Strategic decision: Play or pass?
        if self._should_pass(pass_moves, play_moves, opponent_hand_size, last_move):
            return pass_moves[0]
        
        # Choose best play move using conservative strategy
        return self._choose_conservative_play(play_moves, opponent_hand_size, last_move)
    
    def _get_opponent_hand_size(self, game_state: Optional[GameState]) -> int:
        if not game_state:
            return 17 # assume full hand...
        
        # Determine which player is opponent
        if game_state.curr_player == 1:
            return sum(game_state.player2_vec)
        else:
            return sum(game_state.player1_vec)
    
    # Decide Whether to Pass or NOT
    def _should_pass(self, pass_moves: List[Move], play_moves: List[Move], opponent_hand_size: int, last_move: Optional[Move]) -> bool:
        # Never pass if no pass moves available
        if not pass_moves: return False
        # Never pass on opening move (no last move)
        if last_move is None: return False
        # Don't pass if opponent has very few cards < 5 (pressure them)
        if opponent_hand_size <= 4: return False
        
        # Pass if we can only beat with high-value cards and opponent has many cards
        if opponent_hand_size > 10:
            high_value_moves = [m for m in play_moves if m.primary_rank in self.high_value_ranks]
            if len(high_value_moves) == len(play_moves):  # Only high-value moves available
                return True
        
        # Pass if we can only beat with bombs/rockets and it's not urgent
        powerful_moves = [m for m in play_moves if m.combo_type in [COMBO_BOMB, COMBO_ROCKET]]
        if len(powerful_moves) == len(play_moves) and opponent_hand_size > 8:
            return True
        
        # Pass sometimes to be unpredictable (20% chance when opponent has many cards)
        if opponent_hand_size > 12:
            import random
            return random.random() < 0.2
        
        return False
    
    # Choose the Most Conservative Play
    def _choose_conservative_play(self, play_moves: List[Move], opponent_hand_size: int, last_move: Optional[Move]) -> Move:
        # Separate moves by power level
        regular_moves = []
        bomb_moves = []
        rocket_moves = []
        
        for move in play_moves:
            if move.combo_type == COMBO_ROCKET:
                rocket_moves.append(move)
            elif move.combo_type == COMBO_BOMB:
                bomb_moves.append(move)
            else:
                regular_moves.append(move)
        
        # Conservative priority: regular > bomb > rocket
        # Only use powerful moves when necessary
        
        if regular_moves:
            return self._choose_best_regular_move(regular_moves, opponent_hand_size)
        elif bomb_moves and opponent_hand_size <= 6:  # Use bombs when opponent is close to winning
            return self._choose_best_bomb(bomb_moves)
        elif rocket_moves:  # Last resort
            return rocket_moves[0]
        else:
            # Fallback - shouldn't happen
            return play_moves[0]
    
    # Choose Best Reg Move w Conservative Criteria
    def _choose_best_regular_move(self, regular_moves: List[Move], opponent_hand_size: int) -> Move:
        # Separate by combo type
        singles = [m for m in regular_moves if m.combo_type == COMBO_SINGLE]
        pairs = [m for m in regular_moves if m.combo_type == COMBO_PAIR]
        triples = [m for m in regular_moves if m.combo_type == COMBO_TRIPLE]
        straights = [m for m in regular_moves if m.combo_type == COMBO_STRAIGHT]
        
        # Conservative preference: singles > pairs > triples > straights
        # (Singles preserve more flexibility)
        
        preferred_moves = singles or pairs or triples or straights
        if not preferred_moves:
            preferred_moves = regular_moves
        
        # Among preferred moves, choose lowest rank
        def conservative_priority(move: Move) -> tuple:
            # 1. avoid high-value cards
            rank_val = RANK_VALUES.get(move.primary_rank, 999)
            is_high_value = move.primary_rank in self.high_value_ranks
            
            # 2. prefer fewer cards (more conservative)
            card_count = move.count
            
            # 3. actual rank value
            return (is_high_value, card_count, rank_val)
        
        preferred_moves.sort(key=conservative_priority)
        return preferred_moves[0]
    
    # Lowest Val Bomb
    def _choose_best_bomb(self, bomb_moves: List[Move]) -> Move:
        def bomb_priority(move: Move) -> int:
            return RANK_VALUES.get(move.primary_rank, 999)
        
        bomb_moves.sort(key=bomb_priority)
        return bomb_moves[0]

