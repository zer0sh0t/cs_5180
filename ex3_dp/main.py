import numpy as np
import matplotlib.pyplot as plt
from env import *
from algorithms import *
import matplotlib.pyplot as plt

def main():
    gridworld = Gridworld5x5()

    gamma = 0.9
    theta = 0.001

    # policy = np.ones((gridworld.rows, gridworld.cols, gridworld.action_space)) / gridworld.action_space
    # V = policy_evaluation(policy, gamma, theta, gridworld.transitions, gridworld.rows, gridworld.cols, True)

    # V, policy = value_iteration(gamma, theta, gridworld.transitions, gridworld.rows, gridworld.cols)
    V, policy = policy_iteration(gamma, theta, gridworld.transitions, gridworld.rows, gridworld.cols)

    # V, policy = policy_iteration_6(jcr, gamma, theta)
    # policy = np.array(policy).transpose().tolist()

    print(V)
    print()

    plt.imshow(V)
    plt.colorbar()

    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            plt.text(j, i, str(round(V[i, j], 1)), ha='center', va='center', color='w')

    plt.show()
    
    for p in policy:
        print(p)

    plt.imshow(np.zeros((5, 5)))
    # plt.colorbar()

    act_dict = {0: "l", 1: "d", 2: "r", 3: "u"}
    
    for i in range(5):
        for j in range(5):
            text = ""
            for a in policy[i][j]:
                text += f"{act_dict[a]} "
            
            print(text)
            plt.text(j, i, str(text), ha='center', va='center', color='w')

    plt.show()

if __name__ == "__main__":
    main()
