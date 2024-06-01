import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from algorithms import *
from policy import *
from env import *

def plot_surface(func, usable_ace, q):
    p_values = range(12, 22)
    d_values = range(1, 11)
    v = np.zeros((len(p_values), len(d_values)))

    for ip_, p in enumerate(p_values):
        for id_, d in enumerate(d_values):
            if q == "3a":
                v[id_, ip_] = func[(p, d, usable_ace)]
            elif q == "3b1":
                v[id_, ip_] = max(func[(p, d, usable_ace)])
            elif q == "3b2":
                v[id_, ip_] = func((p, d, usable_ace))

    if q == "3b2":
        i2l = {1: "hit", 0: "stick"}
        im = plt.imshow(v, cmap="gray", extent=[12, 21, 10, 1])
        plt.xlabel("Player sum")
        plt.ylabel("Dealer showing")

        values = np.unique(v.ravel())
        colors = [im.cmap(im.norm(value)) for value in values]
        patches = [mpatches.Patch(color=colors[i], label=f"{i2l[i]}".format(l=values[i])) for i in range(len(values))]
        plt.legend(handles=patches)
        plt.show()

    else:
        plt.imshow(v, cmap="RdGy", vmin=-1, vmax=1, extent=[12, 21, 10, 1])

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # X, Y = np.meshgrid(p_values, d_values)

        # ax.plot_surface(X, Y, v)
        # ax.set_xlabel("Player Sum")
        # ax.set_ylabel("Dealer Showing")

        # if usable_ace:
        #     ax.set_title("Usable Ace")
        # else:
        #     ax.set_title("No Usable Ace")

        plt.xlabel("Player Sum")
        plt.ylabel("Dealer Showing")
        
        if usable_ace:
            plt.title("Usable Ace")
        else:
            plt.title("No Usable Ace")

        plt.legend()
        plt.show()

if __name__ == "__main__":
    # 3
    # env = gym.make("Blackjack-v1", sab=True)
    # n_eps = 800000
    # gamma = 1

    # 3a
    # V = on_policy_mc_evaluation(env, default_blackjack_policy, n_eps, gamma)
    # print(V.keys())
    
    # plot_surface(V, False, "3a")
    # plot_surface(V, True, "3a")

    # 3b
    # Q, policy = on_policy_mc_control_es(env, n_eps, gamma)

    # plot_surface(Q, True, "3b1")
    # plot_surface(Q, False, "3b1")

    # plot_surface(policy, True, "3b2")
    # plot_surface(policy, False, "3b2")

    # 4
    goal_pos = (10, 10)
    env = FourRoomsEnv(goal_pos=goal_pos)
    n_trials = 5
    n_eps = 100
    gamma = 0.99
    # epsilon = 0.1

    agents = [0.1, 0.02, 0]

    trial_returns = []
    for t in range(n_trials):
        returns = []

        for i, epsilon in enumerate(agents):
            r = on_policy_mc_control_epsilon_soft(env, n_eps, gamma, epsilon)
            returns.append(r)

        trial_returns.append(returns)

    trial_returns = np.array(trial_returns)
    # Transpose the arrays to have shape (agents, trials, episodes)
    trial_returns = np.transpose(trial_returns, (1, 0, 2))

    for i, epsilon in enumerate(agents):
        avg_trial_return = np.mean(trial_returns[i], axis=0)
        std_error = 1.96 * np.std(trial_returns[i]) / np.sqrt(n_trials)

        plt.plot(avg_trial_return, label=f"ε={epsilon}")
        plt.fill_between(np.arange(0, len(avg_trial_return)), avg_trial_return - std_error, avg_trial_return + std_error, alpha=0.2)
    
    plt.xlabel("episodes")
    plt.ylabel("average return")
    plt.title(f"average return vs. episodes for {n_trials} trials") 
    plt.legend()
    plt.show()
