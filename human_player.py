import gym_connect_four


class Player():

    player_code : int


    def __init__(self, player_code : int):

        self.player_code = player_code


    def get_movement(self, env : gym_connect_four.ConnectFourEnv):

        while (True):

            try:

                play = input('Your play: ')

                if (play == 'z'):

                    env.undo_moves(2)
                    env.render()
                    print('Game history:', env.play_history())
                    continue

                if (env.is_valid_action(int(play))):

                    return int(play)
                
                raise Exception()
            
            except:

                print('Invalid input!')