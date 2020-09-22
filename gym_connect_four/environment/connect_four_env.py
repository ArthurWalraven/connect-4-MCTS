from enum import Enum, unique
from typing import Tuple, NamedTuple, Optional

import gym
import numpy as np

import numba


class ConnectFourEnv(gym.Env):
    """
    Description:
        ConnectFour game environment

    Observation:
        Type: Discreet(board.shape) (default value is (6,7))

    Actions:
        Type: Discreet(board.shape[1]) (default value is 7)
        Num     Action
        x       Column in which to insert next token (0-6 if the number of columns is 7)

    Reward:
        Reward is 0 for every step.
        If there are no other further steps possible, 0 and termination will occur
        If whites (player code 1) win reward is 1 and termination will occur
        If blacks (player code -1) win reward is -1 and termination will occur
        If it is an invalid move a warning message is prompted and no changes occur

    Starting State:
        Whites to move, empty board (all zeros)

    Episode Termination:
        No more spaces left for pieces (draw)
        Four equal pieces aligned horizontally, vertically or diagonally
    """

    metadata = {'render.modes': ['console']}

    
    action_space : gym.spaces.Discrete
    observation_space : gym.spaces.Discrete
    _reward : np.float
    _is_win : bool
    _is_draw : bool
    _current_player : np.int8
    _board : np.ndarray
    _board_shape : tuple
    _play_history : list

    def __init__(self, board_shape=(6, 7)):
        super(ConnectFourEnv, self).__init__()

        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=board_shape, dtype=np.int8)
        self.action_space = gym.spaces.Discrete(board_shape[1])

        self._reward = np.float(0)
        self._is_win = False
        self._is_draw = False
        self._current_player = np.int8(1)
        self._board = np.zeros(shape=board_shape, dtype=np.int8)
        self._board_shape = board_shape # Using '_board.shape' causes pylint false errors
        self._play_history = []


    def step(self, action: np.uint8) -> Tuple[np.ndarray, np.float, bool, dict]:
        
        # Invalid actions are ignored
        if ((not self.is_valid_action(action)) or self.is_final_state()):
            
            print('WARNING: Ignoring invalid action', action)
            return self._board.copy(), np.float(0), False, {'action performed' : None}
        
        # Perform action
        _insert_value_at_column(value=self._current_player, column=action, board=self._board)
        
        # This order is important
        self._is_draw = _check_for_draw(self._board)
        self._is_win = _check_for_win(self._board)
        self._reward = self._calculate_reward()
        self._play_history.append(action)
        self._current_player *= np.int8(-1)


        return self._board.copy(), self.get_reward(), self.is_final_state(), {'action performed' : action}


    def get_board(self) -> np.ndarray:
        
        return self._board.copy()


    def get_play_history(self) -> list:
        
        return self._play_history.copy()


    def get_move_count(self) -> int:
        
        return len(self._play_history)


    def get_current_player_code(self) -> np.int8:
        
        return self._current_player
    

    def get_player_code_for_whites(self) -> np.int8:

        return np.int8(1)


    def get_player_code_for_blacks(self) -> np.int8:

        return np.int8(-1)


    def reset(self, move_to_return_to : int = 0) -> np.ndarray:

        assert move_to_return_to >= 0
        assert move_to_return_to <= len(self._play_history)
        
        self.undo_moves(len(self._play_history) - move_to_return_to)

        self._is_win = _check_for_win(self._board)
        self._is_draw = _check_for_draw(self._board)
        self._reward = self._calculate_reward()
        
        return self.get_board()
    

    def undo_moves(self, how_many : int = 1) -> None:

        assert how_many >= 0
        assert how_many <= len(self._play_history)


        _remove_values_from_columns(np.array(self._play_history[len(self._play_history) - how_many :], dtype=np.int8), self._board)
        
        del self._play_history[len(self._play_history) - how_many :]
        self._current_player = np.int8(1) if ((len(self._play_history) & 0b1) == 0) else np.int8(-1)
        self._is_win = False if (how_many > 0) else _check_for_win(self._board)
        self._is_draw = False if (how_many > 0) else _check_for_draw(self._board)
        self._reward = np.float(0) if (how_many > 0) else self._calculate_reward()


    def render(self, mode: str = 'console') -> None:
        
        if mode != 'console':
            
            raise gym.error.UnsupportedMode()
            
        drawings = {
            1 : '⬤',
            -1 : '◯',
            0 : ' '
        }

        def print_board_line(line):

            print('║ ', end='')
            for j in range(line.shape[0] - 1):
                print(drawings[line[j]], end='')
                print('  │ ', end='')
            print(drawings[line[-1]], ' ║')


        print('╔' + ('════╤' * (self._board_shape[1] - 1)) + '════╗')
        for i in range(self._board_shape[0] - 1):

            print_board_line(self._board[i, :])
            print('╟' + ('────┼' * (self._board_shape[1] - 1)) + '────╢')
        
        print_board_line(self._board[-1, :])
        print('╚' + ('════╧' * (self._board_shape[1] - 1)) + '════╝')

        
        # Indicate available moves
        for i in range(self._board_shape[1]):
            
            print(' {:2}  '.format(i if i in _available_moves(self._board) else ' '), end='')


        # Indicate next player or final state
        if (self.is_win_state()):
        
            print(' Player', drawings[self._current_player * np.int8(-1)], ' wins!')
        
        elif (self.is_draw_state()):
        
            print(' Draw!')
        
        else:
        
            print(' →', drawings[self._current_player])


    def is_valid_action(self, action: int) -> bool:
        
        return ((self._board[0][action] == 0) and (not self.is_win_state()))


    def is_win_state(self) -> bool:

        return self._is_win

    
    def is_draw_state(self) -> bool:

        return self._is_draw


    def is_final_state(self) -> bool:

        return (self.is_draw_state() or self.is_win_state())


    def get_available_moves(self) -> np.ndarray:

        return _available_moves(self._board)
    

    def get_reward(self) -> np.float:
        
        return self._reward


    def _calculate_reward(self) -> np.float:

        if (self.is_win_state()):

            return self._current_player
                
        return np.float(0)



@numba.jit(nopython=True)
def _insert_value_at_column(value : np.int8, column : np.int, board : np.ndarray) -> None:

    target_column = board[:, column]
    index_of_last_zero = np.where(target_column == np.int8(0))[0][-1]
    target_column[index_of_last_zero] = value


@numba.jit(nopython=True)
def _remove_values_from_columns(list_of_columns : np.ndarray, board : np.ndarray) -> None:

    for i in range(list_of_columns.size):

        column_index = list_of_columns[-i-1]
        column = board[:, column_index]
        index_first_nonzero = np.nonzero(column)[0][0]
        column[index_first_nonzero] = np.int8(0)


@numba.jit(nopython=True)
def _available_moves(board : np.ndarray) -> np.ndarray:

    return np.where(board[0, :] == 0)[0]


@numba.jit(nopython=True)
def _check_for_draw(board : np.ndarray) -> bool:

    return (np.count_nonzero(board[0, :]) == board.shape[1])


@numba.jit(nopython=True)
def _check_for_win(board : np.ndarray) -> bool:

    # Test rows

    for i in range(board.shape[0]):

        for j in range(board.shape[1] - 3):
        
            sum_of_four_elements = np.sum(board[i, j : j + 4])
        
            if (np.abs(sum_of_four_elements) == 4):
        
                return True


    # Test columns (lines in the transposed board)

    transposed_board = board.transpose()
    
    for i in range(board.shape[1]):
    
        for j in range(board.shape[0] - 3):
    
            sum_of_four_elements = np.sum(transposed_board[i, j : j + 4])
    
            if (np.abs(sum_of_four_elements) == 4):
    
                return True


    # Test diagonals

    mirrored_board = np.fliplr(board)

    for i in range(board.shape[0] - 3):
    
        for j in range(board.shape[1] - 3):
    
            sum_of_four_elements = np.sum(np.diag(board[i :, j :])[:4])
    
            if (np.abs(sum_of_four_elements) == 4):
    
                return True

            sum_of_four_elements = np.sum(np.diag(mirrored_board[i :, j :])[:4])
    
            if (np.abs(sum_of_four_elements) == 4):
    
                return True


    return False