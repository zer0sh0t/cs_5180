import numpy as np
from typing import Tuple


class BanditEnv:
    """Multi-armed bandit environment"""

    def __init__(self, k: int) -> None:
        """__init__.

        Args:
            k (int): number of arms/bandits
        """
        self.k = k

    def reset(self) -> None:
        """Resets the mean payout/reward of each arm.
        This function should be called at least once after __init__()
        """
        # Initialize means of each arm distributed according to standard normal
        self.means = np.random.normal(size=self.k)

    def step(self, action: int) -> Tuple[float, int]:
        """Take one step in env (pull one arm) and observe reward

        Args:
            action (int): index of arm to pull
        """
        # calculate reward of arm given by action
        if self.means is None:
            raise ValueError("You must call reset() before taking any steps.")

        # Calculate reward based on the chosen arm (action)
        reward = np.random.normal(loc=self.means[action], scale=1.0)

        return reward
