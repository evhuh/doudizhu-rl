# doudizhu/action_space.py

import sys
import os
from typing import List, Optional, Dict
# from collections import defaultdict # FOR TESTING
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.insert(0, project_root)

from doudizhu.utils import (RANKS, COMBO_SINGLE, COMBO_PAIR, COMBO_TRIPLE, COMBO_STRAIGHT, COMBO_BOMB, COMBO_ROCKET, COMBO_PASS)
from doudizhu.game_engine import Move



class ActionSpace:
    def __init__(self):
        self.actions = self._generate_fixed_actions()
        self.action_to_idx = {self._move_to_key(move): i for i, move in enumerate(self.actions)}
         
    # Main Generating Func
    def _generate_fixed_actions(self) -> List[Move]:
        actions = []
        
        # Action 0: PASS
        actions.append(Move(COMBO_PASS))
        # Singles (3-JOKER)
        for rank in RANKS:
            actions.append(Move(COMBO_SINGLE, rank, 1))
        # Pairs (3-2) 
        for rank in RANKS[:-2]:
            actions.append(Move(COMBO_PAIR, rank, 2))
        # Triples (3-2)
        for rank in RANKS[:-2]:
            actions.append(Move(COMBO_TRIPLE, rank, 3))
        # Bombs (only 3-2)
        for rank in RANKS[:-2]:
            actions.append(Move(COMBO_BOMB, rank, 4))
        # Rocket
        actions.append(Move(COMBO_ROCKET, None, 2, ['joker', 'JOKER']))
        # Actions 60+: All possible straights (5-12 cards, starting 3-8)
        for start_idx in range(8): # 3-10 start
            for length in range(5, 13): # min = 5, max = 12 cards
                if start_idx + length > 12: # A: highest last card
                    break
                straight_ranks = RANKS[start_idx:start_idx + length]
                actions.append(Move(COMBO_STRAIGHT, straight_ranks[0], length, straight_ranks))
        
        return actions

    # Convert Move to unique string key for dictionary lookup
    def _move_to_key(self, move: Move) -> str:
        if move.combo_type == COMBO_PASS:
            return "pass"
        elif move.combo_type == COMBO_SINGLE:
            return f"single_{move.primary_rank}"
        elif move.combo_type == COMBO_PAIR:
            return f"pair_{move.primary_rank}"
        elif move.combo_type == COMBO_TRIPLE:
            return f"triple_{move.primary_rank}"
        elif move.combo_type == COMBO_BOMB:
            return f"bomb_{move.primary_rank}"
        elif move.combo_type == COMBO_ROCKET:
            return "rocket"
        elif move.combo_type == COMBO_STRAIGHT:
            return f"straight_{'_'.join(move.ranks)}"
        else:
            return f"unknown_{move.combo_type}"
        
    # Convert Move Obj to Fixed Action Index
    def move_to_action(self, move: Move) -> Optional[int]:
        key = self._move_to_key(move)
        return self.action_to_idx.get(key)
    
    # Convert action index to Move object
    def action_to_move(self, action_idx: int) -> Move:
        if action_idx >= len(self.actions):
            raise ValueError(f"Action index {action_idx} out of range [0, {len(self.actions)-1}]")
        return self.actions[action_idx]
    
    # Get count of each combo type in action space (for debugging)
    def get_action_breakdown(self) -> Dict[str, int]:
        breakdown = {}
        for action in self.actions:
            combo_type = action.combo_type
            breakdown[combo_type] = breakdown.get(combo_type, 0) + 1
        return breakdown
    
    # Print detailed information about the action space
    def print_action_space_info(self):
        print(f"Total actions: {len(self.actions)}")
        print(f"Action 0: {self.actions[0]}")
        print(f"Action 1: {self.actions[1]}")
        print("Action breakdown:")
        breakdown = self.get_action_breakdown()
        for combo_type, count in breakdown.items():
            print(f"  {combo_type}: {count}")




# TESTING ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# def test():
#     action_space = ActionSpace()
    
#     print(f"Total Actions: {len(action_space.actions)}")
#     print()
    
#     # Group actions by combo type
#     actions_by_type = defaultdict(list)
#     for i, action in enumerate(action_space.actions):
#         actions_by_type[action.combo_type].append((i, action))
    
#     # Print each category
#     for combo_type in [COMBO_PASS, COMBO_SINGLE, COMBO_PAIR, COMBO_TRIPLE, COMBO_BOMB, COMBO_ROCKET, COMBO_STRAIGHT]:
#         if combo_type in actions_by_type:
#             actions = actions_by_type[combo_type]
#             print(f"{combo_type.upper()} ({len(actions)} actions):")
            
#             if combo_type == COMBO_STRAIGHT:
#                 straights_by_length = defaultdict(list)
#                 for i, action in actions:
#                     length = len(action.ranks)
#                     straights_by_length[length].append((i, action))
                
#                 for length in sorted(straights_by_length.keys()):
#                     straight_actions = straights_by_length[length]
#                     print(f"  {length}-card straights ({len(straight_actions)}):")
#                     for i, action in straight_actions:
#                         ranks_str = '-'.join(action.ranks)
#                         print(f"    Action {i:2d}: {ranks_str}")
#                     print()
#             else:
#                 # Print all actions for non-straights
#                 for i, action in actions:
#                     if combo_type == COMBO_PASS:
#                         print(f"  Action {i:2d}: PASS")
#                     elif combo_type == COMBO_ROCKET:
#                         print(f"  Action {i:2d}: Rocket (joker + JOKER)")
#                     else:
#                         print(f"  Action {i:2d}: {combo_type} {action.primary_rank}")
#             print()
    
#     print()
    
#     return action_space

# if __name__ == "__main__":
#     test()

