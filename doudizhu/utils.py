# doudizhu/utils.py

import random
from typing import List, Tuple



# Game Basics
RANKS = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2', 'joker', 'JOKER']
RANK_VALUES = {rank: i for i, rank in enumerate(RANKS)}
SUITS = ['hearts', 'diamonds', 'clubs', 'spades']

# Plays
COMBO_PASS = 'pass'
COMBO_SINGLE = 'single'
COMBO_PAIR = 'pair'
COMBO_TRIPLE = 'triple'
COMBO_STRAIGHT = 'straight'
COMBO_BOMB = 'bomb'
COMBO_ROCKET = 'rocket'

# Game Constants
HAND_SIZE = 17
NUM_RANKS = len(RANKS)

# Type Defs
Card = Tuple[str, str] # (rank, suit)
Hand = List[Card] # list of individual cards # dealing and display
CountVector = List[int] # ct of ea rank # game logic and RL state


# Create Std 54-card Deck
def create_deck() -> List[Card]:
    deck = []
    # add regular cards
    for rank in RANKS[:-2]:
        for suit in SUITS:
            deck.append((rank, suit))
    # add jokers
    deck.append(('joker', None))
    deck.append(('JOKER', None))
    return deck

def shuffle_deck(deck: List[Card]) -> List[Card]:
    random.shuffle(deck)
    return deck

# Deal Out Cards
# UPDATE when necessary for more players
def deal_hands(deck: List[Card]) -> Tuple[Hand, Hand]:
    if len(deck) < HAND_SIZE * 2:
        raise ValueError(f"Not enough cards in deck. Need {HAND_SIZE * 2}, got {len(deck)}")
    
    hand1 = deck[:HAND_SIZE]
    hand2 = deck[HAND_SIZE:HAND_SIZE*2]
    return hand1, hand2

# Convert Hand of Individual Cards to Count-vector Representaiton
def hand_to_counts(hand: Hand) -> CountVector:
    count_vec = [0]*NUM_RANKS
    for rank, suit in hand:
        if rank in RANK_VALUES:
            count_vec[RANK_VALUES[rank]] += 1
        else:
            raise ValueError(f"Unkown rank: {rank}")
        
    return count_vec

# Convert count vector to readable stirng
def counts_to_display(counts: CountVector) -> str:
    display = []
    for i, count in enumerate(counts):
        if count > 0:
            rank = RANKS[i]
            display.append(f"{count}x{rank}")

    if display:
        return ", ".join(display)
    else:
        return "Empty hand"

# Convert each card to readable string
def hand_to_display(hand: Hand) -> str:
    if not hand:
        return "Empty hand"
    
    # Sort hand by rank value for display
    sorted_hand = sorted(hand, key=lambda card: RANK_VALUES[card[0]])

    cards = []
    suit_symbols = {
        'spades': '♠',
        'hearts': '♥', 
        'diamonds': '♦',
        'clubs': '♣'
    }

    for rank, suit in sorted_hand:
        if suit is None:
            cards.append(rank)
        else:
            cards.append(f"{rank}{suit_symbols[suit]}")

    return " ".join(cards)


# Num val for Rank
def get_rank_value(rank: str) -> int:
    return RANK_VALUES.get(rank, -1)

# ? Rank1 > Rank2
def is_higher_rank(rank1: str, rank2: str) -> bool:
    return get_rank_value(rank1) > get_rank_value(rank2)

# Final All POssible Striaght in a Hand
def get_straights_in_hand(counts: CountVector, min_len: int = 5) -> List[List[str]]:
    straights = []

    # Only check ranks 3-A (no 2's or jokers in straights)
    for start_idx in range(12 - min_len + 1):
        for length in range(min_len, 12 - start_idx + 1): # check if can form straight
            can_form = True

            for i in range(start_idx, start_idx + length):
                if counts[i] < 1:
                    can_form = False
                    break
                    
            if can_form:
                straight_ranks = [RANKS[i] for i in range(start_idx, start_idx + length)]
                straights.append(straight_ranks)
    
    return straights

