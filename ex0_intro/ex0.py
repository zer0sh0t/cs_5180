import copy
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable, List
from enum import IntEnum

class Action(IntEnum):
    """Action"""

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

def actions_to_dxdy(action: Action):
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

def dxdy_to_actions(dx, dy):
    mapping = {
        (-1, 0): Action.LEFT,
        (0, -1): Action.DOWN,
        (1, 0): Action.RIGHT,
        (0, 1): Action.UP,
    }
    return mapping[(dx, dy)]

def get_perpendicular_actions(action):
    perp_mapping = {
        Action.LEFT: (Action.UP, Action.DOWN),
        Action.RIGHT: (Action.UP, Action.DOWN),
        Action.UP: (Action.LEFT, Action.RIGHT),
        Action.DOWN: (Action.LEFT, Action.RIGHT),
    }
    return perp_mapping[action]

def is_not_in_walls(state, action, walls):
    next_state = (state[0] + actions_to_dxdy(action)[0], state[1] + actions_to_dxdy(action)[1])
    return next_state not in walls

def reset():
    """Return agent to start state"""
    return (0, 0)

# Q1
def simulate(state: Tuple[int, int], action: Action, debug):
    """Simulate function for Four Rooms environment

    Implements the transition function p(next_state, reward | state, action).
    The general structure of this function is:
        1. If goal was reached, reset agent to start state
        2. Calculate the action taken from selected action (stochastic transition)
        3. Calculate the next state from the action taken (accounting for boundaries/walls)
        4. Calculate the reward

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))
        action (Action): selected action from current agent position (must be of type Action defined above)

    Returns:
        next_state (Tuple[int, int]): next agent position
        reward (float): reward for taking action in state
    """
    # Walls are listed for you
    # Coordinate system is (x, y) where x is the horizontal and y is the vertical direction
    walls = [
        (0, 5),
        (2, 5),
        (3, 5),
        (4, 5),
        (5, 0),
        (5, 2),
        (5, 3),
        (5, 4),
        (5, 5),
        (5, 6),
        (5, 7),
        (5, 9),
        (5, 10),
        (6, 4),
        (7, 4),
        (9, 4),
        (10, 4),
    ]

    # check if goal was reached
    goal_state = (10, 10)
    if state == goal_state:
        print("goal reached!")
        return ((0, 0), 1)

    if debug:
        print(f"current state: {state}")
        print(f"action received: {action}")

    # Modify action_taken so that 10% of the time, it is perpendicular to the selected action
    if np.random.random() < 0.1:
        perpendicular_actions = get_perpendicular_actions(action)
        action_taken = np.random.choice(perpendicular_actions)
    else:
        action_taken = action

    if debug:
        print(f"action selected: {action_taken}")
        print()

    # Calculate the next state based on the action taken
    dx, dy = actions_to_dxdy(action_taken)
    next_x, next_y = (state[0] + dx, state[1] + dy)

    # Check if the next state is within boundaries and is not a wall
    if (0 <= next_x <= 10 and 0 <= next_y <= 10 and (next_x, next_y) not in walls):

        # Check if the next state is the goal state
        if (next_x, next_y) == goal_state:
            return ((0, 0), 1)  # Goal state reached, reset agent to start state and return a reward of 1
        else:
            return ((next_x, next_y), 0)  # Non-goal state with reward 0

    else:
        # If the next state is out of bounds or a wall, stay in the current state with reward 0
        return (state, 0)

# Q2
def manual_policy(state: Tuple[int, int]):
    """A manual policy that queries user for action and returns that action

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        action (Action)
    """
    print(f"current state: {state}")
    init_action = input("please input the action (u, d, l, r): ")
    print()

    action_mapping = {
            'u': Action.UP,
            'd': Action.DOWN,
            'l': Action.LEFT,
            'r': Action.RIGHT,
    }
    action = action_mapping[init_action]
    return action

# Q2
def agent(policy, debug, steps=1000, trials=1):
    """
    An agent that provides actions to the environment (actions are determined by policy), and receives
    next_state and reward from the environment

    The general structure of this function is:
        1. Loop over the number of trials
        2. Loop over total number of steps
        3. While t < steps
            - Get action from policy
            - Take a step in the environment using simulate()
            - Keep track of the reward
        4. Compute cumulative reward of trial

    Args:
        steps (int): steps
        trials (int): trials
        policy: a function that represents the current policy. Agent follows policy for interacting with environment.
            (e.g. policy=manual_policy, policy=random_policy)

    """
    # you can use the following structure and add to it as needed
    trial_rewards = []

    for t in range(trials):
        state = reset()
        cumm_rewards = []

        i = 0
        cumm_reward = 0
        
        while i < steps:
            # select action to take
            action = policy(state)

            # take step in environment using simulate()
            state, reward = simulate(state, action, debug)

            # record the reward
            cumm_reward += reward
            cumm_rewards.append(cumm_reward)

            i += 1

        trial_rewards.append(cumm_rewards)
    return trial_rewards

# Q3
def random_policy(state: Tuple[int, int]):
    """A random policy that returns an action uniformly at random

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        action (Action)
    """
    action_int = np.random.randint(0, 4)
    action = Action(action_int)
    return action

# Q4
def worse_policy(state: Tuple[int, int]):
    """A policy that is worse than the random_policy

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        action (Action)
    """
    action = Action.UP
    return action

def wavefront_planner(state):
    goal_state = (10, 10)
    walls = [
        (0, 5),
        (2, 5),
        (3, 5),
        (4, 5),
        (5, 0),
        (5, 2),
        (5, 3),
        (5, 4),
        (5, 5),
        (5, 6),
        (5, 7),
        (5, 9),
        (5, 10),
        (6, 4),
        (7, 4),
        (9, 4),
        (10, 4),
    ]

    grid = np.zeros((11, 11))
    for (x, y) in walls:
        grid[x, y] = 1

    d = copy.deepcopy(grid)
    d[goal_state] = 2
    L = [goal_state]

    while L != []:
        curr_i, curr_j = L.pop()
        
        for i in range(curr_i-1, curr_i+2):
            for j in range(curr_j-1, curr_j+2):
                if i >= 0 and i <= 10 and j >= 0 and j <= 10:
                    if d[i, j] == 0:
                        d[i, j] = d[curr_i, curr_j] + 1
                        L.append((i, j))
    
    start = copy.deepcopy(state)
    path = [start]
    curr_i, curr_j = start

    while d[curr_i, curr_j] > 2:
        min_dist = float("inf")

        for i in range(curr_i-1, curr_i+2):
            for j in range(curr_j-1, curr_j+2):

                if i >= 0 and i <= 10 and j>= 0 and j <= 10:
                    if d[i, j] < min_dist and d[i, j] >= 2:

                        min_dist = d[i, j]
                        min_ind = (i, j)

        path.append(min_ind)
        curr_i, curr_j = min_ind

    actions_coords = []

    for i in range(len(path)-1):
        x, y = path[i]
        next_x, next_y = path[i+1]
        actions_coords.append((next_x - x, next_y - y))

    simplified_coords = []

    for dx, dy in actions_coords:
        if dx != 0 and dy != 0:
            simplified_coords.append((dx, 0))
            simplified_coords.append((0, dy))
        else:
            simplified_coords.append((dx, dy))

    actions = []
    for dx, dy in simplified_coords:
        actions.append(dxdy_to_actions(dx, dy))

    s = copy.deepcopy(state)
    for i in range(len(actions) - 1):
        a = actions[i]
        if is_not_in_walls(s, a, walls):
            pass
        else:
            actions[i], actions[i + 1] = actions[i + 1], actions[i]

        dx, dy = actions_to_dxdy(a)
        next_s = (s[0] + dx, s[1] + dy)
        s = next_s

    return actions
 
# Q4
def better_policy(state: Tuple[int, int]):
    """A policy that is better than the random_policy

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        action (Action)
    """
    actions = wavefront_planner(state)
    return actions[0] # model predictive controller

def main(steps, trials, debug, plot_option):
    # run code for Q2~Q4 and plot results
    # You may be able to reuse the agent() function for each question
    if plot_option == "manual policy":
        agent(steps=steps, trials=trials, policy=manual_policy, debug=debug)

    else:
        if plot_option == "all policies":
            policies = [better_policy, random_policy, worse_policy]
            c_map = {
                0: "blue",
                1: "black",
                2: "green",
            }
            l_map = {
                0: "Better policy",
                1: "Random policy",
                2: "Worse policy",
            }
                
        elif plot_option == "only random":
            policies = [random_policy]

        for i in range(len(policies)):
            policy = policies[i]

            trial_rewards = agent(steps=steps, trials=trials, policy=policy, debug=debug)
            average_cumm_rewards = np.mean(trial_rewards, axis=0)

            for cumm_rewards in trial_rewards:
                plt.plot(np.arange(0, steps), cumm_rewards, linestyle='dotted')
            
            if len(policies) == 1:
                plt.plot(np.arange(0, steps), average_cumm_rewards, linewidth=2, color="black")
            else:
                c = c_map[i]
                l = l_map[i]
                plt.plot(np.arange(0, steps), average_cumm_rewards, linewidth=2, color=c, label=f"{l}")

        plt.xlabel("Steps")
        plt.ylabel("Cumulative reward")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    # plot options: "only random", "all policies" or "manual policy"
    main(steps=10000, trials=10, debug=False, plot_option="all policies")
