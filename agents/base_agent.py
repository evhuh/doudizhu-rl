# agents/base_agent.py

import sys
import os
from abc import ABC, abstractmethod
from typing import List

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from doudizhu.game_engine import Move, GameState



# Abstract Base Class For All Agents
class Agent(ABC):
    def __init__(self, name: str):
        self.name = name
        self.games_played = 0
        self.games_won = 0
    
    # Choose Move from list of legal moves
    @abstractmethod
    def choose_action(self, legal_moves: List[Move], game_state=None) -> Move:
        pass

    # Call when new game starts, Agents use this for init
    def game_started(self, role: str, player_number: int):
        pass
    
    # Call when game end-- use to track stats
    def game_ended(self, won: bool):
        self.games_played += 1
        if won:
            self.games_won += 1
    
    # Get Agent's win rate (all games played)
    def get_win_rate(self) -> float:
        return self.games_won / max(1, self.games_played)
    
    # Reset game stats
    def reset_stats(self):
        self.games_played = 0
        self.games_won = 0
    
