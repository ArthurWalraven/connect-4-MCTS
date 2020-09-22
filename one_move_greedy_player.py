import random
import numpy as np
import numba

import gym_connect_four


class Player():

    player_code : np.int8


    def __init__(self, player_code : np.int8):

        self.player_code = player_code
    

    def get_movement(self, env : gym_connect_four.ConnectFourEnv, print_simulation_info : bool = True):

        return _get_movement(env._board, np.uint8(self.player_code))


@numba.jit(nopython=True)
def _get_movement(board : np.ndarray, player_code : np.uint8) -> np.int8:
    
    available_moves = gym_connect_four.environment.connect_four_env._available_moves(board)


    for m in available_moves:

        # If I can win in next move, I do it
        gym_connect_four.environment.connect_four_env._insert_value_at_column(value=player_code, column=m, board=board)

        if (gym_connect_four.environment.connect_four_env._check_for_win(board)):

            gym_connect_four.environment.connect_four_env._remove_values_from_columns(np.array([m]), board)
            return m
    
        gym_connect_four.environment.connect_four_env._remove_values_from_columns(np.array([m]), board)
        
        # If the opponent can win in next move, prevent it
        gym_connect_four.environment.connect_four_env._insert_value_at_column(value=-player_code, column=m, board=board)
        
        if (gym_connect_four.environment.connect_four_env._check_for_win(board)):

            gym_connect_four.environment.connect_four_env._remove_values_from_columns(np.array([m]), board)
            return m
    
        gym_connect_four.environment.connect_four_env._remove_values_from_columns(np.array([m]), board)


    return np.random.choice(available_moves)