import time             # time(): To check wen out of time for the current move
import random           # choice()
import numpy as np
import os               # rename(), remove(), path.isfile(): For file handling while training AIs
import copy             # deepcopy(); To copy gym environments
import multiprocessing  # Pool(): to run Player._MCTS() in parallel
from typing import Tuple


import tree
import gym_connect_four

import random_player
import one_move_greedy_player


_RANDOM_PLAYER = random_player.Player(player_code=0)
_ONE_MOVE_GREEDY_PLAYER = one_move_greedy_player.Player(player_code=0)


class _data():
    """ The data hold in the game tree nodes

    This is the core information hold in the nodes of the algorithm tree
    Other information relate to the tree structure itself: parent and children nodes

    Attributes
    ----------
    value : np.float
        the current estimation for the expected reward of the game state
        related to the node

    simulation_count : np.int32
        the number of simulations run from this node. The algorithm (see
        _backpropagation()) ensures this is at least the sum of the children's
        'simulation_count'

    action : np.uint8
        the action LEADING to the related state
        
        That is, if the game is in the state associated with the parent node and
        one performs the action `action`, then the environment will assume the
        state associated with this node
    """

    def __init__(self):
        """ Initialize attributes to default values

        `value` = 0
        `simulation_count` = 0
        `action` = np.uint8(255)

        NOTE: This default value for `action` will be detected if not properly set
        for a node other than the root
        NOTE: In practice, this limits the number of columns in the game to 255
        """
        
        self.value : np.float = np.float(0)
        self.simulation_count : np.int32 = np.int32(0)
        self.action : np.uint8 = np.uint8(255)
    

    def __str__(self):
        # Usefull for tree debugging
        return 'A:{} V:{:.2} #S:{}'.format(self.action, self.value, self.simulation_count)


class Player():
    """ An MCTS based `Connect Four` player

    
    Attributes
    ----------
    tree_root : tree.Node
        the root of the entire game tree if the tree is being held in memory

        In this case, it has `None` as `parent` and `255` as `action`. Otherwise it is `None`
    
    player_code : np.int(8)
        the player code for this player in the `gym_connect_four.ConnectFourEnv`

        The environment has a code for "whites" and for "blacks"
    
    exploration_constant : float
        the exploration constant in UCB formula (used in "selection" steps)
        
        See `select_best_node()`

        The `C` in
            average_reward + C * sqrt(lg(simulation_count_for_parent) / simulation_count)
    
    loaded_file : str
        if a game tree was loaded from a file, holds the name of this file. `None` otherwise
    """

    tree_root : tree.Node
    player_code : np.int8
    exploration_constant : float
    loaded_file : str
    _max_play_time_ms : int
    _use_multiprocessing : bool


    def __init__(self, player_code : np.int8, max_time_sec : float, rollout_policy : str = 'random_greedy', exploration_constant : float = 1.0, load_tree_from_file : str = None, use_multiprocessing : bool = True):
        """ Initialize the Player

        Parameters
        ----------
        player_code : np.int8
            `gym_connect_four.ConnectFourEnv.get_player_code_for_whites()`
            if this the player will move in whites turn and
            `gym_connect_four.ConnectFourEnv.get_player_code_for_blacks()`
            otherwise

        max_time_sec : float
            an action will be decided after this amount of time (in seconds) running the MCTS algorithm

            NOTE: Does not to need to be an integer, e.g., `0.5` is undertood
            NOTE: The actual time used for `step()` method can be higher
                  as this only times the tree search portion of the method
        
        rollout_policy : str
            indicates the policy used in simulations. Either 'random' or 'random_greedy'

            See `_random_play()` and `_short_sighted()` (respectively)
            
        exploration_constant : float
            the exploration constant in UCB formula (used in "selection" steps)
            
            See `select_best_node()`
        
        load_tree_from_file : str
            if provided, tries to load a game tree from the indicated file

            If a file with this name does not exist, nothing happens

        use_multiprocessing : bool
            whether or not to calculate movements in parallel

            If true each possible movement will be calculated in separete trees
            (currently this makes the AI significantly stronger)
            Multiprocessing prevents tree storage and logging information from
            the inner working of the tree search
        """
        
        assert player_code == -1 or player_code == 1, 'Needs to be either `gym_connect_four.ConnectFourEnv.get_player_code_for_whites()` or `gym_connect_four.ConnectFourEnv.get_player_code_for_blacks()`'
        assert max_time_sec >= 0
        assert exploration_constant >= 0
        assert rollout_policy == 'random' or rollout_policy == 'random_greedy'
        assert not use_multiprocessing or not load_tree_from_file, 'Multiprocessing mode does not support tree storage.'
        

        self.player_code = player_code
        self.max_play_time_sec = max_time_sec
        self.exploration_constant = exploration_constant
        self.tree_root = tree.Node(parent=None, data=_data()) if not use_multiprocessing else None
        self._use_multiprocessing = use_multiprocessing
        self.loaded_file = load_tree_from_file

        if (load_tree_from_file):
            
            try:

                self.load_tree(file_name=load_tree_from_file)
                self.loaded_file = load_tree_from_file
            
            except FileNotFoundError:

                print('Starting from fresh tree')
        
        if (rollout_policy == 'random'):

            self._simulation_function = _random_play
        
        elif (rollout_policy == 'random_greedy'):

            self._simulation_function = _short_sighted

    
    def save_tree(self, file_name : str) -> None:

        if (not file_name):

            return


        print("Saving game tree to '{}'... ({} nodes with information about {} simulations)".format(file_name, 1 + tree.count_decendants(self.tree_root), self.tree_root.data.simulation_count))
        
        # Backup just to be sure
        if (os.path.isfile(file_name)):

            print('Creating backup...')
            os.rename(file_name, file_name + '.bkp')
            
        # Write check file to prevent loading a tree made for a player with different color
        print('Writing new file... (This may take a few seconds)')
        with open(file_name + '.check', 'w') as check_file:

            print(self.player_code, file=check_file)
        
        # Save the tree
        tree.pickle_tree(root=self.tree_root, file_name=file_name)

        if (os.path.isfile(file_name + '.bkp')):

            print('Deleting backup...')
            os.remove(file_name + '.bkp')
        
        print('Done.')
    

    def load_tree(self, file_name : str) -> None:

        if (not file_name):

            return


        try:
            with open(file_name + '.check', 'r') as check_file:

                file_player_code = int(check_file.read())

                if (not (self.player_code == file_player_code)):

                    raise Exception('File {} cannot be load since it was created for another player. Current player code: {}. File player code: {}'.format(file_name, self.player_code, file_player_code))
    
        except FileNotFoundError:

            print('File', file_name, 'has no check file. (The related tree was never created)')

            raise FileNotFoundError

        
        print('Loading game tree from file `{}`... (This may take a few seconds)'.format(file_name))
        loaded_tree = tree.unpickle_tree(file_name=file_name)
        print('Done. (Loaded {} nodes with information about {} simulations)'.format(1 + tree.count_decendants(loaded_tree), loaded_tree.data.simulation_count))

        if (loaded_tree.data.simulation_count < self.tree_root.data.simulation_count):

            print('The loaded tree holds information from fewer simulations than the current tree.')
            print('Current tree simulation count:', self.tree_root.data.simulation_count)
            print('Loaded tree simulation count:', loaded_tree.data.simulation_count)
            
            option = input('Are you sure you want to load this tree? (Y/n)')
            
            if ((option != 'y') and (option != 'Y')):

                print('Operation canceled: no changes were made!')
                return
        
        self.tree_root = loaded_tree


    def get_movement(self, env : gym_connect_four.ConnectFourEnv, print_simulation_info : bool = True) -> np.int8:

        assert not env.is_final_state()
        assert self.player_code == env.get_current_player_code(), "I'm player {} and environment's expect a play from {}".format(self.player_code, env.get_current_player_code())
        
        
        if (self._use_multiprocessing):

            return self._get_movement_parallel(env)
        
        return self._get_movement_classic(env, print_simulation_info)
            
    
    def _get_movement_parallel(self, env : gym_connect_four.ConnectFourEnv) -> np.int8:
        
        roots = []
        env_copies = []

        for move in env.get_available_moves():

            roots.append(tree.Node(parent=None, data=_data()))
            roots[-1].children = [tree.Node(parent=roots[-1], data=_data())]
            roots[-1].children[0].data.action = move

            # One environment for each process
            env_copies.append(copy.deepcopy(env))



        with multiprocessing.Pool() as processing_pool:
            
            roots = processing_pool.starmap(self._MCTS, [args for args in zip(roots, env_copies)])


        children = [r.children[0] for r in roots]

        # Greed movement choice
        chosen_child = max(children, key=lambda c: c.data.value)

        # Print confidence in chosen move
        simulation_count = sum([r.data.simulation_count for r in roots])
        expected_reward = np.average([r.data.value for r in roots], weights=[r.data.simulation_count for r in roots])

        confidence = 50 * (expected_reward + 1)
        print('Confidence: {:.1f}%{} '.format(confidence, '!' if confidence > 95 else ' '))
        print('Expected reward from chosen move ({}): {:+.2f}'.format(chosen_child.data.action, chosen_child.data.value))
        print('Decision made based on', simulation_count, 'simulations. All of them performed now.')

        return chosen_child.data.action


    def _get_movement_classic(self, env : gym_connect_four.ConnectFourEnv, print_simulation_info : bool = True) -> np.int8:

        # Find current state in tree
        root = self.tree_root

        original_play_history = env.get_play_history()

        env.reset()
        for move in original_play_history:

            if (not root.children):

                _expand_node(root, env.get_available_moves())
            
            for c in root.children:

                if c.data.action == move:

                    root = c
                    env.step(move)
                    break


        original_simulation_count = root.data.simulation_count
        self._MCTS(root, env, print_simulation_info)

        # Greed movement choice
        chosen_child = max(root.children, key=lambda c: c.data.value)

        # Print confidence in chosen move
        print('\nExpected reward from chosen move ({}): {:+.2f}'.format(chosen_child.data.action, chosen_child.data.value))
        print('Decision made based on', root.data.simulation_count, 'simulations. From these, {} ({:.2f}%) were performed now.'.format((root.data.simulation_count - original_simulation_count), 100 * (root.data.simulation_count - original_simulation_count)/root.data.simulation_count))
        
        return chosen_child.data.action

            
    def _MCTS(self, root : tree.Node, env : gym_connect_four.ConnectFourEnv, print_simulation_info : bool = False) -> tree.Node:

        start_time = time.time()

        original_move_count = env.get_move_count()


        while ((time.time() - start_time) < (self.max_play_time_sec)):

            # Selection
            selected_node, actions_to_reach_selected_node = _select_best_node(root=root, exploration_constant=self.exploration_constant)

            # Follow the steps to reach the desired state
            for move in actions_to_reach_selected_node:

                env.step(move)
            
            # Expansion
            if (not env.is_final_state()):

                # assert not selected_node.children

                selected_node = _expand_node(selected_node, env.get_available_moves())
                env.step(selected_node.data.action)


            # Simulation
            simulation_reward = self._simulation_function(selected_node, env, self.player_code)
            

            # Back propagation
            _back_propagate(selected_node, simulation_reward)


            # Go back to original state
            env.reset(move_to_return_to=original_move_count)


            # Indicate current estimations for move value
            if (print_simulation_info):

                number_of_columns = env.get_board().shape[1]
                action_values = [None for _ in range(number_of_columns)]

                for c in root.children:
                
                    action_values[c.data.action] = c.data.value
                
                print('\r', end='')
                for move in action_values:
                    
                    print('{}.{:<2} '.format('-' if move < 0 else ' ', int(np.trunc(np.abs(move*100) - 1))) if move else '     ', end='')

                confidence = 50 * (root.data.value + 1)
                print(' Confidence: {:.1f}%{} '.format(confidence, '!' if confidence > 95 else ' '), end='')


        return root




def _select_best_node(root : tree.Node, exploration_constant : float) -> Tuple[tree.Node, list]:

    current_node = root
    actions_to_reach_current_node = []
    this_is_my_decision = True

    while (current_node.children):

        children_value = np.array([_evaluate_node(c, exploration_constant, this_is_my_decision) for c in current_node.children])

        # There can be more than one child with the maximum value
        idx_for_best_children = np.where(children_value == np.max(children_value))[0]
        
        # Decide ties randomly
        current_node = current_node.children[np.random.choice(idx_for_best_children)]
        
        actions_to_reach_current_node.append(current_node.data.action)
        
        this_is_my_decision = not this_is_my_decision
    
    
    return current_node, actions_to_reach_current_node


def _evaluate_node(node : tree.Node, exploration_constant : float, this_is_my_decision : bool) -> np.float:

    # Upper Confidence Bound
        
    if (node.data.simulation_count == 0):

        return np.inf
    
    exploration_factor = exploration_constant * np.sqrt(np.log2(node.parent.data.simulation_count) / node.data.simulation_count)
    return exploration_factor + (node.data.value if this_is_my_decision else -node.data.value)


def _expand_node(node : tree.Node, available_actions : np.ndarray) -> tree.Node:

    assert not node.children


    node.children = [tree.Node(parent=node, data=_data()) for move in available_actions]

    for m, c in zip(available_actions, node.children):

        c.data.action = np.uint8(m)


    return random.choice(node.children)


def _random_play(node : tree.Node, env : gym_connect_four.ConnectFourEnv, original_player : int) -> np.float:

    if (env.is_draw_state()):

        node.data.is_final = True
        return 0

    if (env.is_win_state()):

        node.data.is_final = True
        return original_player * env.get_reward()


    done : bool = False
    reward : np.float

    while (not done):

        _, reward, done, _ = env.step(_RANDOM_PLAYER.get_movement(env))
    

    return original_player * reward


def _short_sighted(node : tree.Node, env : gym_connect_four.ConnectFourEnv, original_player : int) -> np.float:

    if (env.is_draw_state()):

        return 0

    if (env.is_win_state()):

        return original_player * env.get_reward()


    done : bool = False

    while (not done):

        _ONE_MOVE_GREEDY_PLAYER.player_code = env.get_current_player_code()
        _, _, done, _ = env.step(_ONE_MOVE_GREEDY_PLAYER.get_movement(env))
    

    return original_player * env.get_reward()


def _back_propagate(node : tree.Node, reward : np.float, weight_of_this_reward : np.int32 = 1) -> None:

    current_node = node

    while (current_node):

        current_node.data.simulation_count += np.int32(weight_of_this_reward)
        current_node.data.value += np.float((reward - weight_of_this_reward * current_node.data.value) / current_node.data.simulation_count)

        current_node = current_node.parent


def train_AIs(env : gym_connect_four.ConnectFourEnv, white_AI_name : str, black_AI_name : str, how_many_seconds_for_each_play : float, how_many_games : int, exploration_constant : float, print_simulation_info : bool, save_tree_after_each_game : bool = True) -> None:

    assert white_AI_name
    assert black_AI_name
    assert how_many_seconds_for_each_play > 0
    assert how_many_games > 0
    assert exploration_constant >= 0


    main_whites = Player(
        player_code=env.get_player_code_for_whites(),
        max_time_sec=how_many_seconds_for_each_play,
        rollout_policy='random_greedy',
        exploration_constant=exploration_constant,
        use_multiprocessing=False,
        load_tree_from_file=white_AI_name + '.tree'
    )
    main_blacks = Player(
        player_code=env.get_player_code_for_blacks(),
        max_time_sec=how_many_seconds_for_each_play,
        rollout_policy='random_greedy',
        exploration_constant=exploration_constant,
        use_multiprocessing=False,
        load_tree_from_file=black_AI_name + '.tree'
    )

    fast_whites = Player(
        player_code=env.get_player_code_for_whites(),
        max_time_sec=0.1,
        rollout_policy='random_greedy',
        exploration_constant=1,
        use_multiprocessing=False
        # load_tree_from_file=black_AI_name + '.tree'
    )

    strong_whites = Player(
        player_code=env.get_player_code_for_whites(),
        max_time_sec=1,
        rollout_policy='random_greedy',
        exploration_constant=1.414213,
        use_multiprocessing=True
        # load_tree_from_file=black_AI_name + '.tree'
    )

    short_sighted_whites = one_move_greedy_player.Player(player_code=env.get_player_code_for_whites())


    # player_whites : gym_connect_four.ConnectFourEnv = short_sighted_whites
    player_whites : gym_connect_four.ConnectFourEnv = fast_whites
    # player_whites : gym_connect_four.ConnectFourEnv = strong_whites
    
    player_blacks : gym_connect_four.ConnectFourEnv = main_blacks


    wins_whites_count : int = 0
    wins_blacks_count : int = 0
    draws_count : int = 0

    for i in range(how_many_games):


        env.reset()

        move : int

        while (not env.is_final_state()):

            print()
            print('Game #: {}/{}'.format((i+1), how_many_games))
            print('⬤  wins:', wins_whites_count, '({:.1f}%)'.format((100 * wins_whites_count/i) if (i > 0) else 0.0), '({})'.format(white_AI_name))
            print('◯  wins:', wins_blacks_count, '({:.1f}%)'.format((100 * wins_blacks_count/i) if (i > 0) else 0.0), '({})'.format(black_AI_name))
            print('Draws:', draws_count, '({:.1f}%)'.format((100 * draws_count / i) if (i > 0) else 0.0))
            print('Time for move: {:.2f}s'.format(how_many_seconds_for_each_play))
            print('\nGame history:', env.get_play_history())
            env.render()
            

            if (env.get_current_player_code() == player_whites.player_code):

                move = player_whites.get_movement(env, print_simulation_info=print_simulation_info)

            elif (env.get_current_player_code() == player_blacks.player_code):

                move = player_blacks.get_movement(env, print_simulation_info=print_simulation_info)
            
            else:

                raise Exception('No known player matches player code {}. Is there a player for whites AND for blacks?'.format(env.get_current_player_code()))


            env.step(move)
        

        # Show final stage
        env.render()
        
        # Update win counts
        if (env.is_draw_state()):

            draws_count += 1
        
        else:

            if (int(env.get_reward()) == player_whites.player_code):

                wins_whites_count += 1
                
            elif (int(env.get_reward()) == player_blacks.player_code):

                wins_blacks_count += 1
            
            else:

                print('WARNING: Error while counting wins!')
        

        if (save_tree_after_each_game):
            
            if (type(player_whites) == Player):
                
                player_whites.save_tree(player_whites.loaded_file)
            
            if (type(player_whites) == Player):
            
                player_blacks.save_tree(player_blacks.loaded_file)


    if (not save_tree_after_each_game):
        
        if (type(player_whites) == Player):
            
            player_whites.save_tree(player_whites.loaded_file)
        
        if (type(player_whites) == Player):
        
            player_blacks.save_tree(player_blacks.loaded_file)
