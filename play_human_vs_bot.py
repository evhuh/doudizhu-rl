# play_human_vs_bot.py

import sys
import os
import random
import time

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path: sys.path.insert(0, project_root)

from doudizhu.utils import COMBO_PASS
from doudizhu.game_engine import DoudizhuGame

from agents import *



class HumanVsBotGame:
    def __init__(self):
        self.available_bots = {
            'random': ('Random Bot', 'Plays completely random legal moves'),
            'greedy': ('Greedy Bot', 'Always plays the smallest valid move'),
            'conservative': ('Conservative Bot', 'Saves powerful moves, prefers singles'),
            'rl': ('Recurrent Learning Bot', 'Trained')
        }
    
    def main_menu(self):
        self._show_welcome()
        
        while True:
            print("\n=====================================================================================================")
            print("\nCHOOSE YOUR OPPONENT:")
            
            for key, (name, description) in self.available_bots.items():
                print(f"  {key}: {name} --> {description}")
            
            print("\nOther options:")
            print("  help: Show game rules")
            print("  quit: Exit the game")
            
            choice = input("\nEnter your choice: ").strip().lower()
            
            if choice == 'quit':
                print("Thanks for playing Dou Dizhu (斗地主)!")
                return
            elif choice == 'help':
                self._show_rules()
                continue
            elif choice in self.available_bots:
                session_result = self._play_game_session(choice)
                if session_result == 'quit':
                    return
            else:
                valid_options = list(self.available_bots.keys()) + ['help', 'quit']
                print(f"ㄨ Invalid choice. Options: {', '.join(valid_options)}")
    
    # Welcome Message
    def _show_welcome(self):
        print("\n" + "˚　　　　✦　　　.　　. 　 ˚　.　　　　　 . ✦　　　 　˚　　　　 . ★⋆. ࿐࿔ ")
        print("\n" + "　　　.   　　˚　　 　　*　　 　　✦　　　.　　.　　　✦　˚ 　　　　 ˚　.˚　　　　✦　　　.　　. 　 ˚　.　　　　 　　 　　　　   ")
        print("⭐️ WELCOME TO DOU DIZHU (斗地主) ⭐️")
        print("Goal: Be the first to play all your cards!")
        print("Use simple inputs: 'P' to pass, '3,4,5,6,7' for cards")
    
    # Game Rules
    def _show_rules(self):
        print("─" * 60)
        print("\n☆ DOU DIZHU RULES ☆")
        print()
        print("• OBJECTIVE:")
        print("    Be the first player to play all your cards!")
        print()
        print("• VALID COMBINATIONS:")
        print("    • Single: Any single card (3, K, A, j, J)")
        print("    • Pair: Two cards of same rank (K,K)")
        print("    • Triple: Three cards of same rank (A,A,A)")
        print("    • Straight: 5+ consecutive ranks (3,4,5,6,7)")
        print("    • Bomb: Four cards of same rank (9,9,9,9)")
        print("    • Rocket: Both jokers together (j,J)")
        print()
        print("• BEATING MOVES:")
        print("    • Same type, higher rank beats lower rank")
        print("    • Bombs beat all regular combinations")
        print("    • Rocket beats everything (including bombs)")
        print("    • You can always pass (P)")
        print()
        print("• CARD RANKING (low → high):")
        print("    3 < 4 < 5 < 6 < 7 < 8 < 9 < 10 < J < Q < K < A < 2")
        print("    j (small joker) < J (big joker)")
        print()
        print("• ROLES:")
        print("    • Landlord: Goes first, tries to empty hand")
        print("    • Farmer: Tries to beat and outlast the landlord")
        print()
        print("• INPUT TIPS:")
        print("    • Order doesn't matter: 7,3,5,4,6 = 3,4,5,6,7")
        print("    • Spaces optional: '3,4,5' = '3, 4, 5'")
        print("    • Use 'P' to pass, 'help' for help, 'quit' to exit")
        print("─" * 60)
    
    # Bot Instance Based on Type
    def _create_bot(self, bot_type: str):
        if bot_type == 'random':
            return RandomAgent(seed=random.randint(1, 10000))
        elif bot_type == 'greedy':
            return GreedyAgent()
        elif bot_type == 'conservative':
            return ConservativeAgent()
        elif bot_type == 'rl':
            return RLAgent(deterministic=True)
        else:
            raise ValueError(f"Unknown bot type: {bot_type}")
    
    # Play a Session of Games Against Chosen Bot
    def _play_game_session(self, bot_type: str) -> str:
        bot_name, bot_description = self.available_bots[bot_type]

        print(f"\n")
        print(f"𓆝  𓆟  𓆞  𓆝  𓆟  𓆝  𓆟  𓆞  𓆝  𓆟  𓆝  𓆟  𓆞  𓆝  𓆟  𓆝  𓆟  𓆞  𓆝  𓆟  𓆝  𓆟  𓆞  𓆝  𓆟  𓆝  𓆟  𓆞  𓆝  𓆟  𓆝  𓆟  𓆞  𓆝")
        print(f"OPPONENT: {bot_name}")
        print(f"Strategy: {bot_description}")
        print(f"𓆝  𓆟  𓆞  𓆝  𓆟  𓆝  𓆟  𓆞  𓆝  𓆟  𓆝  𓆟  𓆞  𓆝  𓆟  𓆝  𓆟  𓆞  𓆝  𓆟  𓆝  𓆟  𓆞  𓆝  𓆟  𓆝  𓆟  𓆞  𓆝  𓆟  𓆝  𓆟  𓆞  𓆝")
        
        human_agent = HumanAgent()
        bot_agent = self._create_bot(bot_type)
        
        session_stats = {
            'games_played': 0,
            'human_wins': 0,
            'bot_wins': 0
        }
        
        while True:
            # Show session stats if games have been played
            if session_stats['games_played'] > 0:
                win_rate = session_stats['human_wins'] / session_stats['games_played'] * 100
                print(f"\nSession: {session_stats['human_wins']}W-{session_stats['bot_wins']}L ({win_rate:.1f}%)")
            
            print("\n🌟What would you like to do?")
            print("  play (or just press Enter): Start a new game")
            print("  menu: Return to bot selection")
            print("  quit: Exit completely")
            
            choice = input("Choice: ").strip().lower()
            
            if choice in ['quit', 'q']:
                # Show final session summary if games were played
                if session_stats['games_played'] > 0:
                    win_rate = session_stats['human_wins'] / session_stats['games_played'] * 100
                    print(f"\nSESSION COMPLETE!")
                    print(f"Final score vs {bot_name}: {session_stats['human_wins']}-{session_stats['bot_wins']}")
                    print(f"Win rate: {win_rate:.1f}%")
                print("Thanks for playing!")
                return 'quit'
            elif choice in ['menu', 'm']:
                # Show final session summary if games were played
                if session_stats['games_played'] > 0:
                    win_rate = session_stats['human_wins'] / session_stats['games_played'] * 100
                    print(f"\nSESSION COMPLETE!")
                    print(f"Final score vs {bot_name}: {session_stats['human_wins']}-{session_stats['bot_wins']}")
                    print(f"Win rate: {win_rate:.1f}%")
                return 'menu'
            elif choice in ['play', 'p', '']:
                result = self._play_single_game(human_agent, bot_agent)
                session_stats['games_played'] += 1
                
                if result == 'human_won':
                    session_stats['human_wins'] += 1
                elif result == 'bot_won':
                    session_stats['bot_wins'] += 1
                # Continue loop for next game
            else:
                print("ㄨ Please enter 'play', 'menu', or 'quit'")
    
    # Play Single Game and Ret Res
    def _play_single_game(self, human_agent, bot_agent) -> str:
        # Ask for role preference
        print(f"🌟 Choose your role:")
        print("  landlord: You go first")
        print("  farmer: Bot goes first")
        print("  random: Random assignment")
        
        while True:
            role_choice = input("Role (landlord/farmer/random): ").strip().lower()
            if role_choice in ['landlord', 'farmer', 'random', 'l', 'f', 'r']:
                if role_choice in ['l', 'landlord']:
                    role_choice = 'landlord'
                elif role_choice in ['f', 'farmer']:
                    role_choice = 'farmer'
                else:
                    role_choice = 'random'
                break
            print("ㄨ Please enter 'landlord', 'farmer', or 'random'")
        
        # Set up game
        game = DoudizhuGame(rng=random.Random())
        
        if role_choice == 'landlord':
            game.start_new_game(p1_role='landlord', p2_role='farmer')
            human_player = 1
            bot_player = 2
        elif role_choice == 'farmer':
            game.start_new_game(p1_role='farmer', p2_role='landlord')
            human_player = 1
            bot_player = 2
        else:  # random
            roles = ['landlord', 'farmer']
            random.shuffle(roles)
            game.start_new_game(p1_role=roles[0], p2_role=roles[1])
            human_player = 1
            bot_player = 2
        
        # Notify agents
        human_role = game.get_player_role(human_player)
        bot_role = game.get_player_role(bot_player)
        
        human_agent.game_started(human_role, human_player)
        bot_agent.game_started(bot_role, bot_player)
        
        # print(f"\nGAME SETUP COMPLETE! ==================================================")
        # print(f"You: Player {human_player} ({human_role.upper()})")
        # print(f"Bot: Player {bot_player} ({bot_role.upper()})")
        
        if human_role == 'landlord':
            print("You start first!")
        else:
            print("Bot starts first!")
        
        # Game loop
        step_count = 0
        max_steps = 200 # prob overkill
        
        while not game.is_game_over() and step_count < max_steps:
            current_player = game.state.curr_player
            
            if current_player == human_player:
                # Human turn
                legal_moves = game.get_legal_moves()
                move = human_agent.choose_action(legal_moves, game.state)
                success = game.play_move(move)
                
                if not success:
                    print("ㄨ Error: Move application failed!")
                    break
                    
            else:
                # Bot turn
                print(f"\n🤖 {bot_agent.name} is thinking...")
                time.sleep(0.5)
                
                legal_moves = game.get_legal_moves()
                move = bot_agent.choose_action(legal_moves, game.state)
                success = game.play_move(move)
                
                if not success:
                    print("ㄨ Error: Bot move failed!")
                    break
                
                # Show bot's move
                if move.combo_type == COMBO_PASS:
                    print(f"🤖 {bot_agent.name} passed")
                else:
                    print(f"🤖 {bot_agent.name} played: {move}")
                
                # Show updated bot hand size
                bot_hand_size = sum(game.state.player2_vec if bot_player == 2 else game.state.player1_vec)
                print(f"🤖 {bot_agent.name} has {bot_hand_size} cards remaining")
            
            step_count += 1
        
        # Game ended - determine result
        if step_count >= max_steps:
            print("Game ended due to timeout!")
            return 'timeout'
        
        winner = game.get_winner()
        
        if winner == human_player:
            result = 'human_won'
            human_agent.game_ended(True)
            bot_agent.game_ended(False)
        elif winner == bot_player:
            result = 'bot_won'
            human_agent.game_ended(False)
            bot_agent.game_ended(True)
        else:
            result = 'draw'
            print("Game ended in a draw!")
        
        return result

# Entry Pt for Human v Bot Games
def main():
    try:
        game_session = HumanVsBotGame()
        game_session.main_menu()
    except KeyboardInterrupt:
        print("\n\nThanks for playing Dou Dizhu (斗地主)!")
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your game files and try again.")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

