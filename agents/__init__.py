# agents/__init__.py

# Agent implementations for Doudizhu
from .base_agent import Agent 
from .random_agent import RandomAgent # Baseline random play
from .greedy_agent import GreedyAgent # Simple rule-based strategy  
from .conservative_agent import ConservativeAgent # Conservative with plays..
from .human_agent import HumanAgent # Human agent
from .rl_agent import RLAgent

__all__ = ['Agent', 'RandomAgent', 'GreedyAgent', 'HumanAgent', 'ConservativeAgent', 'HumanAgent', 'RLAgent']
