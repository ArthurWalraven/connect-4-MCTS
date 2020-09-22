import os

import gym
import human_player
import one_move_greedy_player
import MCTS
from gym_connect_four import ConnectFourEnv


env : ConnectFourEnv = gym.make("ConnectFour-v0", board_shape=(6,7))

player_whites = None
player_blacks = None


print('Game mode:')
print('\t1 - Human vs. Human;')
print('\t2 - Human vs. Machine;')
print('\t3 - Machine vs. Machine;')
print('\t4 - Let AI train;')
while (True):

    try:

        option = int(input('Option: '))

        if (option == 1):

            player_whites = human_player.Player(player_code=env.get_player_code_for_whites())
            player_blacks = human_player.Player(player_code=env.get_player_code_for_blacks())

            break
        
        elif (option == 2):

            player_whites = human_player.Player(player_code=env.get_player_code_for_whites())
            # player_black = one_move_greedy_player.Player(player_code=env.get_player_code_for_blacks())
            player_blacks = MCTS.Player(player_code=env.get_player_code_for_blacks(), max_time_sec=5, exploration_constant=1.0, rollout_policy='random_greedy', use_multiprocessing=True)
            # player_black = MCTS.Player(player_code=env.get_player_code_for_blacks(), max_time_sec=5, exploration_constant=1.0, load_tree_from_file='b_v3.tree')
            
            break
        
        elif (option == 3):

            player_whites = MCTS.Player(
                player_code=env.get_player_code_for_whites(),
                max_time_sec=2,
                rollout_policy='random_greedy',
                exploration_constant=1,
                # load_tree_from_file='w_v3.tree',
                use_multiprocessing=True
            )
            player_blacks = MCTS.Player(
                player_code=env.get_player_code_for_blacks(),
                max_time_sec=2,
                rollout_policy='random_greedy',
                exploration_constant=1,
                load_tree_from_file='b_v3.tree',
                use_multiprocessing=False
            )
            
            break
        
        elif (option == 4):

            MCTS.train_AIs(
                env=env,
                white_AI_name='w_short_sighted',
                black_AI_name='b_v3',
                how_many_seconds_for_each_play=0.25,
                how_many_games=100,
                exploration_constant=1.414213,  # sqrt(2) ~= 1.414213
                save_tree_after_each_game=False,    # This may take some time if the tree is already big.
                                                    # It's better to only set this flag if the script may be interrupted before finishing all plays.
                print_simulation_info=False
            )

            quit()


        raise ValueError
    
    except ValueError as e:

        print(e.args)
        print('Invalid input?')
        

move : int

while (not env.is_final_state()):

    print()
    print('Game history:', env.get_play_history())
    env.render()

    if (env.get_current_player_code() == player_whites.player_code):

        move = player_whites.get_movement(env)

    elif (env.get_current_player_code() == player_blacks.player_code):

        move = player_blacks.get_movement(env)
    
    else:

        raise Exception('No known player matches player code {}. Is there a player for whites AND for blacks?'.format(env.get_current_player_code()))


    env.step(move)


env.render()

# Write simulations learned
if (type(player_whites) == MCTS.Player):

    player_whites.save_tree(player_whites.loaded_file)

if (type(player_blacks) == MCTS.Player):

    player_blacks.save_tree(player_blacks.loaded_file)
