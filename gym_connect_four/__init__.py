from gym.envs.registration import register
from .environment.connect_four_env import ConnectFourEnv

register(
    id='ConnectFour-v0',
    entry_point='gym_connect_four.environment:ConnectFourEnv',
)