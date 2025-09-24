# agents/greedy_agent.py

# 2 kinds of GREEDY
# (A) Greedy for Conservation: Play fewer cards when possible, save high-val combos for when needed
# (B) Greedy for Reduction: PLaye more cards when possible, get rid of cards faster to win sooner
# TODO: Implement (B)...?

# (A) Greedy: Always pick smalles valid move
# 1. Never pass if can play
# 2. Choose lowest rank that beast last move
# 3. Card am played based on (A) or (B) greedy



import sys
import os
from typing import List, Optional

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.insert(0, project_root)

from base_agent import Agent
from doudizhu.utils import COMBO_PASS, RANK_VALUES
from doudizhu.game_engine import Move, GameState



class GreedyAgent(Agent):
    def __init__(self):
        super().__init__("Greedy Agent")
    
    # Avoid passing, Choose smallest valid move
    def choose_action(self, legal_moves: List[Move], game_state: Optional[GameState] = None) -> Move:
        if not legal_moves:
            raise ValueError("No legal moves avail")
        
        # Separate pass from play moves
        play_moves = [move for move in legal_moves if move.combo_type != COMBO_PASS]
        pass_moves = [move for move in legal_moves if move.combo_type == COMBO_PASS]

        # If can play, play
        if play_moves:
            return self._choose_best_play_move(play_moves)
        # Only pass if no choice
        return pass_moves[0]
        
    
    # > choose_action func
    # From valid play moves, choose smallest
    # 1. Lowest rank val 1st
    # 2. For same rank, prefere singles over fairs
    def _choose_best_play_move(self, play_moves: List[Move]) -> Move:
        # (1) Sort by rank val, (2) Sort by card count
        def move_priority(move: Move) -> tuple:
            # Since rockets have type: None
            if move.primary_rank is None:
                return (999, move.count) # put rockets last in prio
            
            rank_val = RANK_VALUES[move.primary_rank]
            return (rank_val, move.count)
        
        # Sort and ret best (smalelst) move
        play_moves.sort(key=move_priority)
        return play_moves[0]

