# agents/human_agent.py

# Interactive human player that takes input via console

import sys
import os
from typing import List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_agent import Agent
from doudizhu.utils import (COMBO_PASS, COMBO_SINGLE, COMBO_PAIR, COMBO_TRIPLE, COMBO_STRAIGHT, COMBO_BOMB, COMBO_ROCKET, RANK_VALUES, RANKS)
from doudizhu.game_engine import Move, GameState



class HumanAgent(Agent):
    def __init__(self):
        super().__init__("Human Player")
        self._show_intro_once = True
    
    def choose_action(self, legal_moves: List[Move], game_state: Optional[GameState] = None) -> Move:
        # Display current game state
        self._display_game_info(game_state, legal_moves)
        
        while True:
            try:
                user_input = input("\nEnter your move: ").strip()
                
                if user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif user_input.lower() == 'quit':
                    print("Quitting game...")
                    sys.exit(0)
                
                # Parse input and find matching move
                chosen_move = self._parse_input_to_move(user_input, legal_moves)
                
                if chosen_move:
                    print(f"You played: {chosen_move}")
                    return chosen_move
                else:
                    print("Invalid move! Please try again.")
                    print("Tip: Use 'P' to pass or enter cards like '3,4,5,6,7'")
                    
            except KeyboardInterrupt:
                print("\nQuitting game...")
                sys.exit(0)
            except Exception as e:
                print(f"Error processing input: {e}")
                print("Please try again.")

    # Parse User IN and Ret Corresp Move Obj
    def _parse_input_to_move(self, user_input: str, legal_moves: List[Move]) -> Optional[Move]:
        # Handle pass
        if user_input.upper() == 'P':
            pass_moves = [m for m in legal_moves if m.combo_type == COMBO_PASS]
            if pass_moves:
                return pass_moves[0]
            else:
                print("ㄨ Cannot pass - you must beat the last move")
                return None
        
        # Parse as card specification
        return self._parse_cards_to_move(user_input, legal_moves)
    
    # Parse Card Specification (ie. '3,4,5,6,7' to Move Obj)
    def _parse_cards_to_move(self, card_input: str, legal_moves: List[Move]) -> Optional[Move]:
        try:
            # Split and clean input - handle spaces around commas
            card_strs = [c.strip() for c in card_input.split(',')]
            card_strs = [c for c in card_strs if c]  # Remove empty strings
            
            if not card_strs:
                print("ㄨ No cards entered")
                return None
            
            # Validate all cards are valid ranks
            invalid_cards = [c for c in card_strs if c not in RANKS]
            if invalid_cards:
                print(f"ㄨ Invalid card(s): {', '.join(invalid_cards)}")
                print(f"✔ Valid cards: {', '.join(RANKS)}")
                return None
            
            # Create candidate move based on input
            candidate_move = self._cards_to_move(card_strs)
            
            if not candidate_move:
                return None
            
            # Check if this move is in legal moves
            matching_move = self._find_matching_legal_move(candidate_move, legal_moves)
            if matching_move:
                return matching_move
            
            print(f"ㄨ Move not legal: {candidate_move}")
            self._suggest_why_illegal(candidate_move, legal_moves, game_state=None)
            return None
            
        except Exception as e:
            print(f"ㄨ Error parsing cards: {e}")
            return None
    
    # Convert List of Cards String to Move Obj
    def _cards_to_move(self, card_strs: List[str]) -> Optional[Move]:
        if len(card_strs) == 1:
            # Single card
            return Move(COMBO_SINGLE, card_strs[0], 1)
        
        elif len(card_strs) == 2:
            # Pair or rocket
            if set(card_strs) == {'j', 'J'}:
                return Move(COMBO_ROCKET, None, 2, ['j', 'J'])
            elif card_strs[0] == card_strs[1]:
                return Move(COMBO_PAIR, card_strs[0], 2)
            else:
                print("ㄨ Two different cards - not a valid pair or rocket")
                return None
        
        elif len(card_strs) == 3:
            # Triple
            if len(set(card_strs)) == 1:  # All same rank
                return Move(COMBO_TRIPLE, card_strs[0], 3)
            else:
                print("ㄨ Three cards must be same rank for triple")
                return None
        
        elif len(card_strs) == 4:
            # Bomb
            if len(set(card_strs)) == 1:  # All same rank
                return Move(COMBO_BOMB, card_strs[0], 4)
            else:
                print("ㄨ Four cards must be same rank for bomb")
                return None
        
        elif len(card_strs) >= 5:
            # Straight
            # Remove duplicates and sort by rank value
            unique_ranks = list(set(card_strs))
            if len(unique_ranks) != len(card_strs):
                print("ㄨ Straights cannot have duplicate cards")
                return None
            
            # Check for invalid straight cards (2s and jokers not allowed)
            invalid_straight_cards = [r for r in unique_ranks if r in ['2', 'j', 'J']]
            if invalid_straight_cards:
                print(f"ㄨ Cannot use {', '.join(invalid_straight_cards)} in straights")
                return None
            
            # Sort ranks by value for straight validation
            try:
                sorted_ranks = sorted(unique_ranks, key=lambda r: RANK_VALUES[r])
                
                # Check if it forms a valid consecutive straight
                if self._is_valid_straight(sorted_ranks):
                    return Move(COMBO_STRAIGHT, sorted_ranks[0], len(sorted_ranks), sorted_ranks)
                else:
                    print("ㄨ Cards do not form a consecutive straight")
                    print(f"You entered: {', '.join(sorted_ranks)} (sorted)")
                    return None
                    
            except KeyError as e:
                print(f"ㄨ Invalid rank in straight: {e}")
                return None
        
        else:
            print("ㄨ Invalid number of cards")
            return None

    # Check if Sorted ranks Form a Valid Consectutive Straight
    def _is_valid_straight(self, sorted_ranks: List[str]) -> bool:
        if len(sorted_ranks) < 5:
            return False
        
        # Check consecutive values
        for i in range(1, len(sorted_ranks)):
            if RANK_VALUES[sorted_ranks[i]] != RANK_VALUES[sorted_ranks[i-1]] + 1:
                return False
        
        return True
    
    # Find Legal Move that Matches the Candidate Move
    def _find_matching_legal_move(self, candidate_move: Move, legal_moves: List[Move]) -> Optional[Move]:
        for legal_move in legal_moves:
            if self._moves_equivalent(candidate_move, legal_move):
                return legal_move
        return None
    
    # Check if 2 Moves Equiv
    def _moves_equivalent(self, move1: Move, move2: Move) -> bool:
        if move1.combo_type != move2.combo_type:
            return False
        
        if move1.combo_type == COMBO_PASS:
            return True
        elif move1.combo_type == COMBO_ROCKET:
            return True
        elif move1.combo_type == COMBO_STRAIGHT:
            return sorted(move1.ranks) == sorted(move2.ranks)
        else:
            return (move1.primary_rank == move2.primary_rank and 
                   move1.count == move2.count)
    
    # Feedback
    def _suggest_why_illegal(self, candidate_move: Move, legal_moves: List[Move], game_state: Optional[GameState]):
        legal_types = [m.combo_type for m in legal_moves if m.combo_type != COMBO_PASS]
        
        if not legal_types:
            print("💡 You can only pass right now")
            return
        
        if candidate_move.combo_type not in legal_types:
            type_names = {
                COMBO_SINGLE: "single", COMBO_PAIR: "pair", COMBO_TRIPLE: "triple",
                COMBO_STRAIGHT: "straight", COMBO_BOMB: "bomb", COMBO_ROCKET: "rocket"
            }
            legal_type_names = [type_names.get(t, str(t)) for t in set(legal_types)]
            print(f"💡 Current move type required: {', '.join(legal_type_names)}")
        else:
            # Same type but not strong enough
            if candidate_move.combo_type == COMBO_SINGLE:
                min_rank = min(m.primary_rank for m in legal_moves if m.combo_type == COMBO_SINGLE)
                print(f"💡 Single card must be {min_rank} or higher")
            elif candidate_move.combo_type == COMBO_PAIR:
                min_rank = min(m.primary_rank for m in legal_moves if m.combo_type == COMBO_PAIR)
                print(f"💡 Pair must be {min_rank} or higher")
            elif candidate_move.combo_type == COMBO_TRIPLE:
                min_rank = min(m.primary_rank for m in legal_moves if m.combo_type == COMBO_TRIPLE)
                print(f"💡 Triple must be {min_rank} or higher")
            elif candidate_move.combo_type == COMBO_STRAIGHT:
                legal_straights = [m for m in legal_moves if m.combo_type == COMBO_STRAIGHT]
                if legal_straights:
                    min_start = min(RANK_VALUES[m.primary_rank] for m in legal_straights)
                    min_rank = [r for r, v in RANK_VALUES.items() if v == min_start][0]
                    print(f"💡 Straight must start with {min_rank} or higher (same length)")
    

    def _display_game_info(self, game_state: Optional[GameState], legal_moves: List[Move]):
        print("\n───────────────────────- YOUR TURN -──────────────────────")
        
        if game_state:
            # Determine which hand to show (current player's hand)
            if game_state.curr_player == 1:
                hand = game_state.player1_hand
                hand_vec = game_state.player1_vec
                opponent_cards = sum(game_state.player2_vec)
            else:
                hand = game_state.player2_hand
                hand_vec = game_state.player2_vec
                opponent_cards = sum(game_state.player1_vec)
            
            from doudizhu.utils import hand_to_display, counts_to_display
            
            print(f"🃁 Your hand: {hand_to_display(hand)}")
            print(f"Card counts: {counts_to_display(hand_vec)}")
            print(f"Opponent: {opponent_cards} cards left")
            
            if game_state.last_move and game_state.last_move.combo_type != COMBO_PASS:
                print(f"Last move: {game_state.last_move} (beat this or pass)")
            else:
                print("No last move (play anything)")
        
        print("───────────────────────-------------──────────────────────")
    
    # Help Instructions
    def _show_help(self):
        print("\n======== DETAILED HELP ========")
        print()
        print("• PASSING:")
        print("   P                    → Pass your turn")
        print()
        print("• SINGLE CARDS:")
        print("   3                    → Play a 3")
        print("   K                    → Play a King")
        print("   j                    → Play small joker")
        print("   J                    → Play big joker")
        print()
        print("• COMBINATIONS:")
        print("   3,3                  → Pair of 3s")
        print("   K,K,K                → Triple Kings")
        print("   9,9,9,9              → Bomb of 9s")
        print("   3,4,5,6,7            → Straight (any order)")
        print("   j,J                  → Rocket (both jokers)")
        print()
        print("• CARD ORDER:")
        print("   7,3,5,4,6    = 3,4,5,6,7    (order doesn't matter)")
        print("   K, K, K      = K,K,K        (spaces optional)")
        print()
        print("• CARD VALUES (low to high):")
        print("   3 < 4 < 5 < 6 < 7 < 8 < 9 < 10 < J < Q < K < A < 2")
        print("   j (small joker) < J (big joker)")
        print()
        print("• OTHER COMMANDS:")
        print("   help                 → Show this help")
        print("   quit                 → Exit game")
        print("─" * 50)
    
    
    def game_started(self, role: str, player_number: int):
        print(f"\nGAME STARTED! You are Player {player_number} ({role.upper()})")
        if role == 'landlord':
            print("You go first as the landlord!")
        else:
            print("You are the farmer - beat the landlord!")
    
    def game_ended(self, won: bool):
        super().game_ended(won)
        if won:
            print("⋆˙⟡ CONGRATS! You won! ⋆˙⟡")
        else:
            print("You lost this round...better luck next time!")
        
        win_rate = self.get_win_rate() * 100
        print(f"Your WR: {win_rate:.1f}% ({self.games_won}/{self.games_played})")

