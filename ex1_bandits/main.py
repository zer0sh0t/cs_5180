from env import BanditEnv
from tqdm import trange
import matplotlib.pyplot as plt
from agent import *

def q4(k: int, num_samples: int):
    """Q4

    Structure:
        1. Create multi-armed bandit env
        2. Pull each arm `num_samples` times and record the rewards
        3. Plot the rewards (e.g. violinplot, stripplot)

    Args:
        k (int): Number of arms in bandit environment
        num_samples (int): number of samples to take for each arm
    """

    env = BanditEnv(k=k)
    env.reset()

    # Initialize a list to record rewards for each arm
    arm_rewards = [[] for _ in range(k)]

    # Pull each arm `num_samples` times and record the rewards
    for arm in range(k):
        for _ in range(num_samples):
            reward = env.step(arm)
            arm_rewards[arm].append(reward)

    # Plot the rewards using a violinplot
    plt.figure(figsize=(10, 6))
    plt.violinplot(arm_rewards, showmeans=True)# , showmedians=True)
    plt.xlabel("Arm")
    plt.ylabel("Rewards")
    plt.title(f"Rewards Distribution for {k}-Armed Bandit (Each arm pulled {num_samples} times)")
    plt.show()


def q6(k: int, trials: int, steps: int):
    """Q6

    Implement epsilon greedy bandit agents with an initial estimate of 0

    Args:
        k (int): number of arms in bandit environment
        trials (int): number of trials
        steps (int): total number of steps for each trial
    """

    # Initialize the environment
    env = BanditEnv(k=k)
    env.reset()

    # Initialize epsilon values for different epsilon-greedy agents
    epsilons = [0.0, 0.01, 0.1]
    
    # Initialize lists to store rewards for each agent
    trial_rewards = [[] for _ in range(trials)]
    trial_opt_actions = [[] for _ in range(trials)]
    trial_expected_rewards = []

    # Loop over trials
    for t in trange(trials, desc="Trials"):
        # Reset environment after every trial
        env.reset()

        # Initialize epsilon-greedy agents with different epsilon values
        agents = [EpsilonGreedy(k=k, init=0, epsilon=epsilon) for epsilon in epsilons]
        for agent in agents:
            agent.reset()

        # Initialize arrays to store rewards for each step and agent
        # avg_reward = [[] for _ in agents]
        rewards = [[] for _ in agents]

        optimal_action_pcts = [[] for _ in agents]
        optimal_action_counts =[[] for _ in agents]

        for step in range(steps):
            for i, agent in enumerate(agents):
                # Choose an action using epsilon-greedy strategy
                action = agent.choose_action()

                optimal_action = np.argmax(env.means)
                is_optimal_action = (action == optimal_action)

                # Take a step in the environment
                reward = env.step(action)
                rewards[i].append(reward)

                optimal_action_counts[i].append(is_optimal_action)

                # Update the agent's Q-values
                agent.update(action, reward)

                # Record the reward for this step and agent
                # avg_reward[i].append(np.mean(rewards[i]))
                optimal_action_pcts[i].append(np.mean(optimal_action_counts[i]) * 100)

        # Store the rewards for this trial
        # trial_rewards[t] = avg_reward
        trial_rewards[t] = rewards
        trial_opt_actions[t] = optimal_action_pcts
        trial_expected_rewards.append(np.max(env.means))

    # avg_reward = np.array(avg_reward)
    trial_rewards = np.array(trial_rewards)
    trial_rewards = np.transpose(trial_rewards, (1, 0, 2))

    trial_opt_actions = np.array(trial_opt_actions)
    trial_opt_actions = np.transpose(trial_opt_actions, (1, 0, 2))
    # print(avg_reward.shape, trial_rewards.shape, trial_opt_actions.shape)

    # Calculate the upper bound based on true expected rewards q*(a)
    max_expected_reward = np.mean(trial_expected_rewards)
    ub_std_error = 1.96 * np.std(trial_expected_rewards) / np.sqrt(trials)
    upper_bound = max_expected_reward * np.ones(steps)
    # print(upper_bound.shape, ub_std_error.shape)

    # Plot the results
    plt.figure(figsize=(10, 6))
    for i, epsilon in enumerate(epsilons):
        avg_trial_reward = np.mean(trial_rewards[i], axis=0)
        std_error = 1.96 * np.std(trial_rewards[i]) / np.sqrt(trials)

        plt.plot(range(steps), avg_trial_reward, label=f"epsilon={epsilon}") 
        plt.fill_between(range(steps), avg_trial_reward - std_error, avg_trial_reward + std_error, alpha=0.2)

    plt.plot(range(steps), upper_bound, label="Upper Bound", linestyle="--")
    plt.fill_between(range(steps), upper_bound - ub_std_error, upper_bound + ub_std_error, alpha=0.2)
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.title(f"Average Reward vs. Steps for {trials} Trials")
    plt.legend()

    plt.figure(figsize=(10, 6))
    for i, epsilon in enumerate(epsilons):
        opt_action_pct = np.mean(trial_opt_actions[i], axis=0)
        std_error = 1.96 * np.std(trial_opt_actions[i]) / np.sqrt(trials)

        plt.plot(range(steps), opt_action_pct, label=f"epsilon={epsilon}")
        plt.fill_between(range(steps), opt_action_pct - std_error, opt_action_pct + std_error, alpha=0.2)

    plt.xlabel("Steps")
    plt.ylabel("% Optimal action")
    plt.title(f"% Optimal actions vs. Steps for {trials} trials")
    plt.legend()

    plt.show()

def q7(k: int, trials: int, steps: int):
    """Q7

    Compare epsilon greedy bandit agents and UCB agents

    Args:
        k (int): number of arms in bandit environment
        trials (int): number of trials
        steps (int): total number of steps for each trial
    """
    # Initialize the environment
    env = BanditEnv(k=k)
    env.reset()

    # Define different configurations for agents
    agent_configs = [
        {"epsilon": 0.0, "init_value": 0},
        {"epsilon": 0.0, "init_value": 5},
        {"epsilon": 0.1, "init_value": 0},
        {"epsilon": 0.1, "init_value": 5},
        {"ucb_c": 2, "init_value": 0}
    ]

    # Initialize lists to store rewards and % Optimal action for each agent
    trial_rewards = [[] for _ in range(trials)]
    trial_optimal_actions = [[] for _ in range(trials)]
    trial_expected_rewards = []

    # Loop over trials
    for t in trange(trials, desc="Trials"):
        # Reset environment after every trial
        env.reset()

        # Initialize agents with different configurations
        agents = []
        for config in agent_configs:
            if "epsilon" in config:
                agent = EpsilonGreedy(k=k, init=config["init_value"], epsilon=config["epsilon"])
            elif "ucb_c" in config:
                agent = UCB(k=k, init=config["init_value"], c=config["ucb_c"], step_size=0.1)
            agents.append(agent)
        
        for agent in agents:
            agent.reset()

        # Initialize arrays to store rewards and % Optimal action for each step and agent
        # avg_reward = [[] for _ in agents]
        rewards = [[] for _ in agents]
        optimal_action_pcts = [[] for _ in agents]
        optimal_action_counts = [[] for _ in agents]

        for step in range(steps):
            for i, agent in enumerate(agents):
                # Choose an action using the agent's strategy
                action = agent.choose_action()

                optimal_action = np.argmax(env.means)
                is_optimal_action = (action == optimal_action)

                # Take a step in the environment
                reward = env.step(action)
                rewards[i].append(reward)
                optimal_action_counts[i].append(is_optimal_action)

                # Update the agent's Q-values
                agent.update(action, reward)

                # Record the reward and % Optimal action for this step and agent
                # avg_reward[i].append(np.mean(rewards[i]))
                optimal_action_pcts[i].append(np.mean(optimal_action_counts[i]) * 100)

        # Store the rewards and % Optimal action for this trial
        # trial_rewards[t] = avg_reward
        trial_rewards[t] = rewards
        trial_optimal_actions[t] = optimal_action_pcts
        trial_expected_rewards.append(np.max(env.means))

    # Convert lists to numpy arrays
    trial_rewards = np.array(trial_rewards)
    trial_optimal_actions = np.array(trial_optimal_actions)

    # Transpose the arrays to have shape (agents, trials, steps)
    trial_rewards = np.transpose(trial_rewards, (1, 0, 2))
    trial_optimal_actions = np.transpose(trial_optimal_actions, (1, 0, 2))

    # Calculate the upper bound based on true expected rewards q*(a)
    max_expected_reward = np.mean(trial_expected_rewards)
    ub_std_error = 1.96 * np.std(trial_expected_rewards) / np.sqrt(trials)
    upper_bound = max_expected_reward * np.ones(steps)

    # Plot the results
    plt.figure(figsize=(12, 6))
    for i, config in enumerate(agent_configs):  # Exclude UCB
        avg_trial_reward = np.mean(trial_rewards[i], axis=0)
        std_error = 1.96 * np.std(trial_rewards[i]) / np.sqrt(trials)

        try:
            label = f"epsilon={config['epsilon']}, Q1={config['init_value']}"
        except:
            label = f"ucb_c={config['ucb_c']}" 

        plt.plot(range(steps), avg_trial_reward, label=label)
        plt.fill_between(range(steps), avg_trial_reward - std_error, avg_trial_reward + std_error, alpha=0.2)

    # Add the Upper Bound to both plots
    plt.plot(range(steps), upper_bound, label="Upper Bound", linestyle="--")
    plt.fill_between(range(steps), upper_bound - ub_std_error, upper_bound + ub_std_error, alpha=0.2)

    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.title(f"Average Reward vs. Steps for {trials} Trials")
    plt.legend()

    # Plot % Optimal action
    plt.figure(figsize=(12, 6))
    for i, config in enumerate(agent_configs):  # Exclude UCB
        opt_action_pct = np.mean(trial_optimal_actions[i], axis=0)
        std_error = 1.96 * np.std(trial_optimal_actions[i]) / np.sqrt(trials)

        try:
            label = f"epsilon={config['epsilon']}, Q1={config['init_value']}"
        except:
            label = f"ucb_c={config['ucb_c']}"

        plt.plot(range(steps), opt_action_pct, label=label)
        plt.fill_between(range(steps), opt_action_pct - std_error, opt_action_pct + std_error, alpha=0.2)

    plt.xlabel("Steps")
    plt.ylabel("% Optimal action")
    plt.title(f"% Optimal actions vs. Steps for {trials} trials")
    plt.legend()

    plt.show()

def main():
    # run code for all questions
    # q4(10, 2000)
    q6(10, 2000, 2000)
    # q7(10, 100, 2000)

if __name__ == "__main__":
    main()
