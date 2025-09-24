# doudizhu/game_engine.py

# GAME LOGIC
# - represent move
# - track game state
# - apply a move, end tricks, and detect wins
#   - compute legal actions from a hand and last move
# - run a full game w roles, dealing, move history

import sys
import os
import random
from typing import List, Optional

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.insert(0, project_root)

from doudizhu.utils import *



# Single move in Dou Dizhu
class Move:
    def __init__(self, combo_type: str, primary_rank: Optional[str]=None, count: int=0, ranks: List[str]=None):
        self.combo_type = combo_type
        self.count = count # num cards in move
        self.primary_rank = primary_rank # rank being played
        self.ranks = ranks or [] # for straights

    def __str__(self):
        if self.combo_type == COMBO_PASS: return "PASS"
        elif self.combo_type == COMBO_SINGLE: return f"Single {self.primary_rank}"
        elif self.combo_type == COMBO_PAIR: return f"Pair of {self.primary_rank}s"
        elif self.combo_type == COMBO_TRIPLE: return f"Triple {self.primary_rank}s"
        elif self.combo_type == COMBO_STRAIGHT: return f"Straight {'-'.join(self.ranks)}"
        elif self.combo_type == COMBO_BOMB: return f"Bomb {self.primary_rank}s"
        elif self.combo_type == COMBO_ROCKET: return f"Rocket (joker pair)"
        else: return f"Unknown move: {self.combo_type}"
    
    # check card rnakings
    def __eq__(self, other):
        if not isinstance(other, Move): return False
        return (
                self.combo_type == other.combo_type and
                self.count == other.count and
                self.primary_rank == other.primary_rank and
                self.ranks == other.ranks
                )


# Curr State of Game
class GameState:
    def __init__(self):
        self.player1_vec: CountVector = []
        self.player2_vec: CountVector = []
        self.curr_player: int = 1 # 1 or 2
        self.last_move: Optional[Move] = None
        self.game_over: bool = False
        self.winner: Optional[int] = None

        # Role tracking
        self.player1_role: str = None  # landlord or farmer
        self.player2_role: str = None  # ""

        # Display
        self.player1_hand: Hand = []
        self.player2_hand: Hand = []

        # To Handle double passing
        self.consecutive_passes = 0
        self.last_non_pass_player = None

    # Get Curr Player's hand (count vec)
    def get_curr_vec(self) -> CountVector:
        return self.player1_vec if self.curr_player == 1 else self.player2_vec
    
    # Get Curr Player's hand
    def get_curr_hand(self) -> Hand:
        return self.player1_hand if self.curr_player == 1 else self.player2_hand
    
    # Switch to Other Player
    def switch_player(self):
        self.curr_player = 2 if self.curr_player == 1 else 1


# Check if Move Can Beat Last Move Played
# > get_legal_moves func
def can_beat_move(move: Move, last_move: Move) -> bool:
    # No last move or last move was pass - any move is valid
    if last_move is None or last_move.combo_type == COMBO_PASS:
        return True

    # Rocket beats everything
    if move.combo_type == COMBO_ROCKET:
        return True
    
    # Nothing beats rocket
    if last_move.combo_type == COMBO_ROCKET:
        return False

    # Bombs beat everything except rockets and higher bombs
    if move.combo_type == COMBO_BOMB:
        if last_move.combo_type == COMBO_BOMB:
            result = is_higher_rank(move.primary_rank, last_move.primary_rank)
            return result
        else:
            return True
    
    # Nothing beats bombs except rockets and higher bombs
    if last_move.combo_type == COMBO_BOMB:
        return False

    # For all other combos, types must match exactly
    if move.combo_type != last_move.combo_type:
        return False

    
    # Same combo type - compare based on specific rules
    if move.combo_type == COMBO_STRAIGHT:
        if len(move.ranks) != len(last_move.ranks):
            return False
        result = is_higher_rank(move.primary_rank, last_move.primary_rank)
        return result
    
    elif move.combo_type in [COMBO_SINGLE, COMBO_PAIR, COMBO_TRIPLE]:
        result = is_higher_rank(move.primary_rank, last_move.primary_rank)
        return result
    
    return False


# Get All Legal Moves for a given (1) Hand AND (2) Last Move Played
#  > apply_move func
def get_legal_moves(vec: CountVector, last_move: Optional[Move] = None) -> List[Move]:
    legal_moves = []

    # COMBO_PASS always legal, except for 1st move of game
    if last_move is not None and last_move.combo_type != COMBO_PASS:
        legal_moves.append(Move(COMBO_PASS))
    
    # Singles
    for i, count in enumerate(vec):
        if count > 0:
            primary_rank = RANKS[i]
            single_move = Move(COMBO_SINGLE, primary_rank, 1)
            # Legal when no last move (player before passed), or if single beats last single
            if last_move is None or can_beat_move(single_move, last_move):
                legal_moves.append(single_move)
    
    # Pairs
    for i, count in enumerate(vec):
        if count >= 2:
            primary_rank = RANKS[i]
            pair_move = Move(COMBO_PAIR, primary_rank, 2)
            # Legal when no last move, or if pair beats last pair
            if last_move is None or can_beat_move(pair_move, last_move):
                legal_moves.append(pair_move)
    
    # Triples
    for i, count in enumerate(vec):
        if count >= 3:
            primary_rank = RANKS[i]
            triple_move = Move(COMBO_TRIPLE, primary_rank, 3)
            if last_move is None or can_beat_move(triple_move, last_move):
                legal_moves.append(triple_move)
    
    # Bombs
    for i, count in enumerate(vec):
        if count == 4:
            primary_rank = RANKS[i]
            bomb_move = Move(COMBO_BOMB, primary_rank, 4)
            if last_move is None or can_beat_move(bomb_move, last_move):
                legal_moves.append(bomb_move)

    # Rocket (like UK arugula XD)
    if vec[13] >= 1 and vec[14] >= 1: # joker and JOKER
        rocket_move = Move(COMBO_ROCKET, None, 2, ['joker', 'JOKER'])
        if last_move is None or can_beat_move(rocket_move, last_move):
            legal_moves.append(rocket_move)

    # Straights
    poss_straights = get_straights_in_hand(vec)
    # print(f"DEBUG: Found {len(poss_straights)} possible straights")
    for straight_ranks in poss_straights:
        primary_rank_str = straight_ranks[0]
        straight_move = Move(COMBO_STRAIGHT, straight_ranks[0], len(straight_ranks), straight_ranks)
        if last_move is None or can_beat_move(straight_move, last_move):
            legal_moves.append(straight_move)
    
    return legal_moves


# Apply a Move to the Game State, ret T if move was valid and applied
def apply_move(game_state: GameState, move: Move) -> bool:
    if game_state.game_over: return False

    # (1) Get Legal Moves for Curr Player (2) Check if Move Legal
    legal_moves = get_legal_moves(game_state.get_curr_vec(), game_state.last_move)
    if move not in legal_moves: return False
    
    # Check if Need to Clear Last Move
    if move.combo_type == COMBO_PASS:
        game_state.consecutive_passes += 1

        # In 2-player game, if both passed, winner starts fresh
        if game_state.consecutive_passes >= 1:
            # reset for new trick
            game_state.last_move = None
            game_state.consecutive_passes = 0
            # force curr player to be winner of last trick
            game_state.curr_player = game_state.last_non_pass_player
            game_state.last_non_pass_player = None
            return True 
    else: 
        # Non-pass move
        game_state.consecutive_passes = 0
        game_state.last_non_pass_player = game_state.curr_player

    # Apply Move
    if move.combo_type != COMBO_PASS:
        # Remove Cards from:
        curr_vec = game_state.get_curr_vec()
        curr_hand = game_state.get_curr_hand()

        if move.combo_type in [COMBO_SINGLE, COMBO_PAIR, COMBO_TRIPLE, COMBO_BOMB]:
            rank_index = RANK_VALUES[move.primary_rank]

            # (1) vec representation
            if curr_vec[rank_index] < move.count: return False
            curr_vec[rank_index] -= move.count

            # (2) Actual hand (DISPLAY)
            cards_removed = 0
            i = 0
            while i < len(curr_hand) and cards_removed < move.count:
                if curr_hand[i][0] == move.primary_rank:
                    curr_hand.pop(i)
                    cards_removed += 1
                else:
                    i += 1

        elif move.combo_type == COMBO_ROCKET:
            # (1) remove both jokers from vec rep
            if curr_vec[13] < 1 or curr_vec[14] <1:
                return False
            curr_vec[13] -= 1
            curr_vec[14] -= 1

            # (2) Actual hand (DISPLAY)
            joker_removed = False
            JOKER_removed = False
            i = 0
            while i < len(curr_hand) and (not joker_removed or not JOKER_removed):
                if curr_hand[i][0] == 'joker' and not joker_removed:
                    curr_hand.pop(i)
                    joker_removed = True
                elif curr_hand[i][0] == 'JOKER' and not JOKER_removed:
                    curr_hand.pop(i)
                    JOKER_removed = True
                else:
                    i += 1
        
        elif move.combo_type == COMBO_STRAIGHT:
            # (1) remove all cards in straight from vec rep
            for rank in move.ranks:
                rank_index = RANK_VALUES[rank]
                if curr_vec[rank_index] < 1: return False
                curr_vec[rank_index] -= 1

                # (2) Actual hand (DSIPALY)
                for i, (card_rank, suit) in enumerate(curr_hand):
                    if card_rank == rank:
                        curr_hand.pop(i)
                        break

        
        # Update Last Move
        game_state.last_move = move
        # Check for Win Cond
        if sum(curr_vec) == 0:
            game_state.game_over = True
            game_state.winner = game_state.curr_player
            return True
    
    game_state.switch_player()
    return True


# Get Reward for a Players
def get_game_reward(winner: int, player: int) -> int:
    if winner == player: return 1
    else: return -1

# Get Complete Game Res w Rewards for Both Players
def get_game_res(winner: int) -> dict:
    return {
        'winner': winner,
        'player1_reward': get_game_reward(winner, 1),
        'player2_reward': get_game_reward(winner, 2),
        'game_over': True
    }



# MAIN
# Manages a Game of Dou Dizhu
class DoudizhuGame:
    def __init__(self, rng=None):
        self.rng = rng if rng is not None else random.Random()
        self.state = GameState()
        self.move_history = []
    
    # Start new game w clean role assign: p1 & p2 roles are either (a) landlord, (b) farmer (c) random
    def start_new_game(self, p1_role: str = 'random', p2_role: str = 'random'):
        # Reset Game State
        self.state = GameState()
        self.move_history = []
        
        # Create and Deal Cards
        deck = create_deck()
        shuffled_deck = shuffle_deck(deck)
        hand1, hand2 = deal_hands(shuffled_deck)
        
        # Set up Game State
        self.state.player1_hand = hand1
        self.state.player2_hand = hand2
        self.state.player1_vec = hand_to_counts(hand1)
        self.state.player2_vec = hand_to_counts(hand2)
        
        # Resolve role assignments
        final_p1_role = self._resolve_role(p1_role)
        final_p2_role = self._resolve_role(p2_role)
        
        # Ensure exactly 1 landlord and 1 farmer
        if final_p1_role == final_p2_role:
            # Both got same role, flip one randomly
            if random.choice([True, False]):
                final_p1_role = 'farmer' if final_p1_role == 'landlord' else 'landlord'
            else:
                final_p2_role = 'farmer' if final_p2_role == 'landlord' else 'landlord'
        
        # Set Roles in Game State
        self.state.player1_role = final_p1_role
        self.state.player2_role = final_p2_role
        
        # Landlord goes 1st always
        if final_p1_role == 'landlord': # P1 ll
            self.state.curr_player = 1
            landlord_player = 1
            farmer_player = 2
        else: # P2 is ll
            self.state.curr_player = 2
            landlord_player = 2
            farmer_player = 1

        # print(f"Game started! Player {landlord_player} is landlord and goes first")
        # print(f"Player {farmer_player} is farmer")
        # self.print_game_state()
    
    # Resolve Role String --> Actual Role
    def _resolve_role(self, role: str) -> str:
        if role.lower() == 'landlord':
            return 'landlord'
        elif role.lower() == 'farmer':
            return 'farmer'
        elif role.lower() == 'random':
            return self.rng.choice(['landlord', 'farmer'])
        else:
            raise ValueError(f"Invalid role: {role}. Must be 'landlord', 'farmer', or 'random'")
    
    # Play a Move; ret whether if was successful
    def play_move(self, move: Move) -> bool:
        # Save curr player before apply_move switches it
        player_who_moved = self.state.curr_player
        
        if apply_move(self.state, move):
            self.move_history.append((player_who_moved, move))
            print(f"Player {player_who_moved} played: {move}")
            return True
        else:
            print(f"Invalid move: {move}")
            return False
        
    
    # Validation Method
    def validate_state(self) -> bool:
        # CHECK: Handsize match vectors
        p1_vec_sum = sum(self.state.player1_vec)
        p1_hand_size = len(self.state.player1_hand)
        p2_vec_sum = sum(self.state.player2_vec)
        p2_hand_size = len(self.state.player2_hand)
        
        if p1_vec_sum != p1_hand_size:
            print(f"Player 1 inconsistency: vec={p1_vec_sum}, hand={p1_hand_size}")
            return False
        if p2_vec_sum != p2_hand_size:
            print(f"Player 2 inconsistency: vec={p2_vec_sum}, hand={p2_hand_size}")
            return False
            
        # CHECK: No negative counts
        if any(count < 0 for count in self.state.player1_vec + self.state.player2_vec):
            print("Negative card counts detected")
            return False
            
        return True
    

    # Get Legal Moves for Curr Player
    def get_legal_moves(self) -> List[Move]: 
        return get_legal_moves(self.state.get_curr_vec(), self.state.last_move)
    
    # Check if Game Over
    def is_game_over(self) -> bool: 
        return self.state.game_over
    
    # If Game Over, Get Winner
    def get_winner(self) -> Optional[int]: 
        return self.state.winner
    
    # Get player's role
    def get_player_role(self, player: int) -> str:
        return self.state.player1_role if player == 1 else self.state.player2_role
    
    # If GAME OVER: Get Game Res w Rewards (RL)
    def get_game_result(self) -> Optional[dict]:
        if not self.state.game_over: 
            return None
        return get_game_res(self.state.winner)
    
    # After GAME OVER: Get Reward for Specific Winner (RL)
    def get_player_reward(self, player: int) -> int:
        if not self.state.game_over: 
            raise ValueError("Can't get reward. Game not over!")
        return get_game_reward(self.state.winner, player)
    

    # Print Curr Game State
    def print_game_state(self):
        print("\n" + "="*50)
        print(f"Player 1 ({self.state.player1_role}): {hand_to_display(self.state.player1_hand)}")
        print(f"Player 1 vec: {counts_to_display(self.state.player1_vec)}")
        print(f"Player 2 ({self.state.player2_role}): {hand_to_display(self.state.player2_hand)}")
        print(f"Player 2 vec: {counts_to_display(self.state.player2_vec)}")
        print(f"Current player: Player {self.state.curr_player}")
        print(f"Last move: {self.state.last_move if self.state.last_move else 'None'}")
        
        if self.state.game_over:
            result = self.get_game_result()
            print(f"GAME OVER! Player {self.state.winner} wins!")
            print(f"Final Scores - P1: {result['player1_reward']}, P2: {result['player2_reward']}")
        else:
            legal_moves = self.get_legal_moves()
            print(f"Legal moves for Player {self.state.curr_player}: {[str(m) for m in legal_moves]}")

