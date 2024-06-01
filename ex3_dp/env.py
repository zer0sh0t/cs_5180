from scipy.stats import poisson
import numpy as np
from enum import IntEnum
from typing import Tuple

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
    # mapping = {
    #     Action.LEFT: (-1, 0),
    #     Action.DOWN: (0, -1),
    #     Action.RIGHT: (1, 0),
    #     Action.UP: (0, 1),
    # }

    mapping = {
         Action.LEFT: (-1, 0),
         Action.DOWN: (0, 1),
         Action.RIGHT: (1, 0),
         Action.UP: (0, -1),
     }
    return mapping[action]


class Gridworld5x5:
    """5x5 Gridworld"""

    def __init__(self) -> None:
        """
        State: (x, y) coordinates

        Actions: See class(Action).
        """
        self.rows = 5
        self.cols = 5
        self.state_space = [
            (x, y) for x in range(0, self.rows) for y in range(0, self.cols)
        ]
        self.action_space = len(Action)

        # set the locations of A and B, the next locations, and their rewards
        self.A = (1, 0)
        self.A_prime = (1, 4)
        self.A_reward = 10
        self.B = (3, 0)
        self.B_prime = (3, 2)
        self.B_reward = 5
        
    def transitions(
        self, state: Tuple, action: Action
    ) -> Tuple[Tuple[int, int], float]:
        """Get transitions from given (state, action) pair.

        Note that this is the 4-argument transition version p(s',r|s,a).
        This particular environment has deterministic transitions

        Args:
            state (Tuple): state
            action (Action): action

        Returns:
            next_state: Tuple[int, int]
            reward: float
        """

        next_state = None
        reward = None

        # Check if the current state is A or B and return the corresponding next state and reward
        if state == self.A:
            next_state = self.A_prime
            reward = self.A_reward

        elif state == self.B:
            next_state = self.B_prime
            reward = self.B_reward

        else:
            dx, dy = actions_to_dxdy(action)
            x, y = state

            next_x = x + dx
            next_y = y + dy

            if (0 <= next_x <= self.cols - 1) and (0 <= next_y <= self.rows - 1):
                reward = 0
            else:
                next_x = x
                next_y = y
                reward = -1
            
            next_state = (next_x, next_y)

        return next_state, reward

    def expected_return(
        self, V, state: Tuple[int, int], action: Action, gamma: float
    ) -> float:
        """Compute the expected_return for all transitions from the (s,a) pair, i.e. do a 1-step Bellman backup.

        Args:
            V (np.ndarray): list of state values (length = number of states)
            state (Tuple[int, int]): state
            action (Action): action
            gamma (float): discount factor

        Returns:
            ret (float): the expected return
        """

        next_state, reward = self.transitions(state, action)
        # compute the expected return
        ret = reward + gamma * V[next_state[1], next_state[0]]

        return ret


class JacksCarRental:
    def __init__(self, modified: bool = False) -> None:
        """JacksCarRental

        Args:
           modified (bool): False = original problem Q6a, True = modified problem for Q6b

        State: tuple of (# cars at location A, # cars at location B)

        Action (int): -5 to +5
            Positive if moving cars from location A to B
            Negative if moving cars from location B to A
        """
        self.modified = modified

        self.action_space = list(range(-5, 6))

        self.rent_reward = 10
        self.move_cost = 2

        # For modified problem
        self.overflow_cars = 10
        self.overflow_cost = 4

        # Rent and return Poisson process parameters
        # Save as an array for each location (Loc A, Loc B)
        self.rent = [poisson(3), poisson(4)]
        self.return_ = [poisson(3), poisson(2)]

        # Max number of cars at end of day
        self.max_cars_end = 20
        # Max number of cars at start of day
        self.max_cars_start = self.max_cars_end + max(self.action_space)

        self.state_space = [
            (x, y)
            for x in range(0, self.max_cars_end + 1)
            for y in range(0, self.max_cars_end + 1)
        ]

        # Store all possible transitions here as a multi-dimensional array (locA, locB, action, locA', locB')
        # This is the 3-argument transition function p(s'|s,a)
        self.t = np.zeros(
            (
                self.max_cars_end + 1,
                self.max_cars_end + 1,
                len(self.action_space),
                self.max_cars_end + 1,
                self.max_cars_end + 1,
            ),
        )

        # Store all possible rewards (locA, locB, action)
        # This is the reward function r(s,a)
        self.r = np.zeros(
            (self.max_cars_end + 1, self.max_cars_end + 1, len(self.action_space))
        )
        self.precompute_transitions()

    def _open_to_close(self, loc_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the probability of ending the day with s_end \in [0,20] cars given that the location started with s_start \in [0, 20+5] cars.

        Args:
            loc_idx (int): the location index. 0 is for A and 1 is for B. All other values are invalid
        Returns:
            probs (np.ndarray): list of probabilities for all possible combination of s_start and s_end
            rewards (np.ndarray): average rewards for all possible s_start
        """
        probs = np.zeros((self.max_cars_start + 1, self.max_cars_end + 1))
        rewards = np.zeros(self.max_cars_start + 1)

        # for start in range(probs.shape[0]):
        #     # TODO Calculate average rewards.
        #     # For all possible s_start, calculate the probability of renting k cars.
        #     # Be sure to consider the case where business is lost (i.e. renting k > s_start cars)
        #     avg_rent = 0.0
        #     rewards[start] = self.rent_reward * avg_rent

        #     # TODO Calculate probabilities
        #     # Loop over every possible s_end
        #     for end in range(probs.shape[1]):
        #         prob = 0.0
        #         # Since s_start and s_end are specified,
        #         # you must rent a minimum of max(0, start-end)
        #         min_rent = max(0, start - end)

        #         # TODO Loop over all possible rent scenarios and compute probabilities
        #         # Be sure to consider the case where business is lost (i.e. renting k > s_start cars)
        #         for i in range(min_rent, start + 1):
        #             pass

        #         probs[start, end] = prob
        
        for start in range(probs.shape[0]):
            avg_rent = 0.0

            # print(f"start: {start}")

            for rented in range(0, self.max_cars_start+1):
                # print(f"rented: {rented}")

                prob_rent = self.rent[loc_idx].pmf(rented)
                revenue = self.rent_reward * min(rented, start)

                # print(f"prob_rent: {prob_rent}")

                if rented > start:
                    revenue = 0

                # print(f"revenue: {revenue}")
                avg_rent += prob_rent * revenue

                # print(f"avg_rent: {avg_rent}")
                # print()

            # exit()

            rewards[start] = avg_rent

            for end in range(probs.shape[1]):
                prob = 0.0
                min_rent = max(0, start - end)

                # print(f"end: {end}")
                # print(f"min_rent: {min_rent}")

                for rented in range(min_rent, start + 1):
                    returned = rented - min_rent
                    
                    # print(f"rented: {rented}")
                    # print(f"return_: {return_}")
            
                    prob_rent = self.rent[loc_idx].pmf(rented)
                    prob_return = self.return_[loc_idx].pmf(returned)

                    # print(f"prob_rent: {prob_rent}")
                    # print(f"prob_return: {prob_return}")

                    prob += prob_rent * prob_return
                    # print(f"prob: {prob}")
                    # print()

                for rented in range(start+1, self.max_cars_end + 1):
                    returned = rented - min_rent
                    prob_rent = 1 - self.rent[loc_idx].cdf(start - 1)
                    prob_return = self.return_[loc_idx].pmf(returned)
                    prob += prob_rent * prob_return
                
                probs[start, end] = prob
            # exit()

        return probs, rewards

    def _calculate_cost(self, state: Tuple[int, int], action: int) -> float:
        """A helper function to compute the cost of moving cars for a given (state, action)

        Note that you should compute costs differently if this is the modified problem.

        Args:
            state (Tuple[int,int]): state
            action (int): action
        """
        cost = self.move_cost * abs(action)
        return cost

    def _valid_action(self, state: Tuple[int, int], action: int) -> bool:
        """Helper function to check if this action is valid for the given state

        Args:
            state:
            action:
        """
        if state[0] < action or state[1] < -(action):
            return False
        else:
            return True

    def precompute_transitions(self) -> None:
        """Function to precompute the transitions and rewards.

        This function should have been run at least once before calling expected_return().
        You can call this function in __init__() or separately.

        """
        # Calculate open_to_close for each location
        day_probs_A, day_rewards_A = self._open_to_close(0)
        day_probs_B, day_rewards_B = self._open_to_close(1)

        # Perform action first then calculate daytime probabilities
        for locA in range(self.max_cars_end + 1):
            for locB in range(self.max_cars_end + 1):
                for ia, action in enumerate(self.action_space):
                    # Check boundary conditions
                    if not self._valid_action((locA, locB), action):
                        self.t[locA, locB, ia, :, :] = 0
                        self.r[locA, locB, ia] = 0
                    else:
                        # Calculate day rewards from renting
                        # Use day_rewards_A and day_rewards_B and _calculate_cost()
                        day_reward_A = day_rewards_A[locA]
                        day_reward_B = day_rewards_B[locB]
                        self.r[locA, locB, ia] = day_reward_A + day_reward_B - self._calculate_cost((locA, locB), action)

                        # Loop over all combinations of locA_ and locB_
                        for locA_ in range(self.max_cars_end + 1):
                            for locB_ in range(self.max_cars_end + 1):
                                # Calculate transition probabilities
                                # Use the probabilities computed from open_to_close
                                day_prob_A = day_probs_A[locA][locA_]
                                day_prob_B = day_probs_B[locB][locB_]
                                self.t[locA, locB, ia, locA_, locB_] = day_prob_A * day_prob_B

    def expected_return(
        self, V, state: Tuple[int, int], action: Action, gamma: float
    ) -> float:
        """Compute the expected_return for all transitions from the (s,a) pair, i.e. do a 1-step Bellman backup.

        Args:
            V (np.ndarray): list of state values (length = number of states)
            state (Tuple[int, int]): state
            action (Action): action
            gamma (float): discount factor

        Returns:
            ret (float): the expected return
        """

        # compute the expected return
        ret = 0
        for locA_ in range(self.max_cars_end + 1):
            for locB_ in range(self.max_cars_end + 1):
                transition_prob = self.t[state[0], state[1], self.action_space.index(action), locA_, locB_]
                reward = self.r[state[0], state[1], self.action_space.index(action)]
                ret += transition_prob * (reward + gamma * V[locA_, locB_])

        return ret

    def transitions(self, state: Tuple, action: Action) -> np.ndarray:
        """Get transition probabilities for given (state, action) pair.

        Note that this is the 3-argument transition version p(s'|s,a).
        This particular environment has stochastic transitions

        Args:
            state (Tuple): state
            action (Action): action

        Returns:
            probs (np.ndarray): return probabilities for next states. Since transition function is of shape (locA, locB, action, locA', locB'), probs should be of shape (locA', locB')
        """
        return self.t[state[0], state[1], self.action_space.index(action), :, :]

    def rewards(self, state, action) -> float:
        """Reward function r(s,a)

        Args:
            state (Tuple): state
            action (Action): action
        Returns:
            reward: float
        """
        return self.r[state[0], state[1], self.action_space.index(action)]
