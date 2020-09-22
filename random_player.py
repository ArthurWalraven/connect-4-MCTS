import random
import numpy as np

import numba

import gym_connect_four


class Player():

    def __init__(self, player_code : np.int8):

        self.player_code = player_code


    def get_movement(self, env : gym_connect_four.ConnectFourEnv) -> np.uint8:

        return _get_movement(env._board)


@numba.jit(nopython=True)
def _get_movement(board : np.ndarray) -> np.int8:

    return np.random.choice(gym_connect_four.environment.connect_four_env._available_moves(board))