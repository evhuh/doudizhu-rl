# agents/random_agent.py

# Agent that chooses random legal moves

import sys
import os
import random
from typing import List, Optional

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path: sys.path.insert(0, project_root)

from base_agent import Agent
from doudizhu.game_engine import Move, GameState



class RandomAgent(Agent):
    def __init__(self, seed: Optional[int] = None):
        super().__init__("Random Agent")
        self.rng = random.Random(seed)

    # Rand choose
    def choose_action(self, legal_moves: List[Move], game_state: Optional[GameState] = None) -> Move:
        if not legal_moves:
            raise ValueError("No legal moves avail")

        return self.rng.choice(legal_moves)

    def set_seed(self, seed: int):
        self.rng.seed(seed)

