from enum import IntEnum
from typing import Tuple, Optional, List
from gym import Env, spaces
from gym.utils import seeding
from gym.envs.registration import register
import random


def register_env(env_name) -> None:
    """Register custom gym environment so that we can use `gym.make()`

    In your main file, call this function before using `gym.make()` to use the Four Rooms environment.
        register_env()
        env = gym.make('WindyGridWorld-v0')

    There are a couple of ways to create Gym environments of the different variants of Windy Grid World.
    1. Create separate classes for each env and register each env separately.
    2. Create one class that has flags for each variant and register each env separately.

        Example:
        (Original)     register(id="WindyGridWorld-v0", entry_point="env:WindyGridWorldEnv")
        (King's moves) register(id="WindyGridWorldKings-v0", entry_point="env:WindyGridWorldEnv", **kwargs)

        The kwargs will be passed to the entry_point class.

    3. Create one class that has flags for each variant and register env once. You can then call gym.make using kwargs.

        Example:
        (Original)     gym.make("WindyGridWorld-v0")
        (King's moves) gym.make("WindyGridWorld-v0", **kwargs)

        The kwargs will be passed to the __init__() function.

    Choose whichever method you like.
    """
    # TODO
    register(id="WindyGridWorld-v0", entry_point="env:WindyGridWorldEnv",kwargs={'king_move': False,'stochiastic':False})

class Action(IntEnum):
    """Action"""

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


def actions_to_dxdy(action: Action) -> Tuple[int, int]:
    """
    Helper function to map action to changes in x and y coordinates
    Args:
        action (Action): taken action
    Returns:
        dxdy (Tuple[int, int]): Change in x and y coordinates
    """
    mapping = {
        Action.LEFT: (-1, 0),
        Action.DOWN: (0, -1),
        Action.RIGHT: (1, 0),
        Action.UP: (0, 1),
    }
    return mapping[action]

class KingAction(IntEnum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    RIGHTUP = 4
    LEFTUP =5
    DOWNRIGHT = 6
    DOWNLEFT = 7
    STAY = 8

def king_action_to_dxdy(action: KingAction) -> Tuple[int,int]:
    mapping={KingAction.LEFT: (-1,0),
             KingAction.DOWN: (0,-1),
             KingAction.RIGHT: (1,0),
             KingAction.UP: (0,1),
             KingAction.RIGHTUP: (1,1),
             KingAction.LEFTUP: (-1,1),
             KingAction.DOWNRIGHT: (1,-1),
             KingAction.DOWNLEFT: (-1,-1),
             KingAction.STAY: (0,0),
             }
    return mapping[action]

class WindyGridWorldEnv(Env):
    def __init__(self, king_move=False, stochastic=False):
        """Windy grid world gym environment
        This is the template for Q4a. You can use this class or modify it to create the variants for parts c and d.
        """

        # Grid dimensions (x, y)
        self.rows = 10
        self.cols = 7
        self.king_move =  king_move
        self.stoch = stochastic
        # Wind
        # TODO define self.wind as either a dict (keys would be states) or multidimensional array (states correspond to indices)
        #self.wind = [0,0,0,1,1,1,2,2,1,0]
        self.wind = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 1, 9: 0}
        if self.king_move:
            self.action_space = spaces.Discrete(len(KingAction))
        else:
            self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(self.rows), spaces.Discrete(self.cols))
        )

        # Set start_pos and goal_pos
        self.start_pos = (0, 3)
        self.goal_pos = (7, 3)
        self.agent_pos = None

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Fix seed of environment

        In order to make the environment completely reproducible, call this function and seed the action space as well.
            env = gym.make(...)
            env.seed(seed)
            env.action_space.seed(seed)

        This function does not need to be used for this assignment, it is given only for reference.
        """

        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.agent_pos = self.start_pos
        return self.agent_pos

    def step(self, action) -> Tuple[Tuple[int, int], float, bool, dict]:
        """Take one step in the environment.

        Takes in an action and returns the (next state, reward, done, info).
        See https://github.com/openai/gym/blob/master/gym/core.py#L42-L58 foand r more info.

        Args:
            action (Action): an action provided by the agent

        Returns:
            observation (object): agent's observation after taking one step in environment (this would be the next state s')
            reward (float) : reward for this transition
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning). Not used in this assignment.
        """

        if  not self.king_move:
            current_state = self.agent_pos
            goal = self.goal_pos
            action = actions_to_dxdy(action)
            wind_affect = self.wind[current_state[0]]

            if self.stoch:
                stochastic_wind_effect = random.choice([-1, 0, 1])
                wind_affect += stochastic_wind_effect
                
            self.agent_pos = (
            max(0, min(current_state[0] + action[0], self.rows - 1)),  # Ensure within valid range for rows
            max(0, min(current_state[1] + action[1] + wind_affect, self.cols - 1))  # Ensure within valid range for columns
            )

            next_state = self.agent_pos
            if next_state == self.goal_pos:
                done = True
                reward = 0.0
            else:
                done = False
                reward = -1.0
            return self.agent_pos, reward, done, {}

        else:
            current_state = self.agent_pos
            goal = self.goal_pos
            action = king_action_to_dxdy(action)
            wind_affect = self.wind[current_state[0]]

            if self.stoch:
                stochastic_wind_effect = random.choice([-1, 0, 1])
                wind_affect += stochastic_wind_effect

            self.agent_pos =  (
            max(0, min(current_state[0] + action[0], self.rows - 1)),  # Ensure within valid range for rows
            max(0, min(current_state[1] + action[1] + wind_affect, self.cols - 1))  # Ensure within valid range for columns
            )

            next_state = self.agent_pos
            if next_state == self.goal_pos:
                done = True
                reward = 0.0
            else:
                done = False
                reward = -1.0
            
            return self.agent_pos,reward,done,{}
