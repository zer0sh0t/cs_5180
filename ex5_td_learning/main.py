from env import *
from algorithms import *
import matplotlib.pyplot as plt

def plot(env, num_steps, gamma=1, epsilon=0.1, step_size=0.5):
    # SARSA
    episode_count,tag=sarsa(env,num_steps,gamma,epsilon,step_size)
    avg_ep = np.mean(np.array(episode_count),0)
    plt.plot(np.arange(0,num_steps,1),avg_ep,label=tag)
    std_error1 = 1.96*np.std(episode_count)/np.sqrt(10)
    plt.fill_between(np.arange(0,num_steps,1),avg_ep - std_error1,avg_ep + std_error1,alpha=0.4)
 
    # EXP_SARSA
    episode_count,tag=exp_sarsa(env,num_steps,gamma,epsilon,step_size)
    avg_ep = np.mean(np.array(episode_count),0)
    plt.plot(np.arange(0,num_steps,1),avg_ep,label=tag)
    std_error1 = 1.96*np.std(episode_count)/np.sqrt(10)
    plt.fill_between(np.arange(0, num_steps, 1), avg_ep - std_error1,avg_ep + std_error1,alpha=0.4)
   
    # N-Step Sarsa
    episode_count,tag=nstep_sarsa(env,num_steps,gamma,epsilon,step_size)
    avg_ep = np.mean(np.array(episode_count),0)
    plt.plot(np.arange(0,num_steps,1),avg_ep,label=tag)
    std_error1 = 1.96*np.std(episode_count)/np.sqrt(10)
    plt.fill_between(np.arange(0, num_steps, 1), avg_ep - std_error1, avg_ep + std_error1, alpha=0.4)
    
    #MC on Policy
    episode_count,tag=on_policy_mc_control_epsilon_soft(env,num_steps,gamma,epsilon)
    avg_ep = np.mean(np.array(episode_count),0)
    plt.plot(np.arange(0,num_steps,1),avg_ep,label=tag)
    std_error1 = 1.96*np.std(episode_count)/np.sqrt(10)
    plt.fill_between(np.arange(0, num_steps, 1), avg_ep - std_error1, avg_ep + std_error1, alpha=0.4)
    
    #Q-Learning
    episode_count,tag=q_learning(env,num_steps,gamma,epsilon,step_size)
    avg_ep = np.mean(np.array(episode_count),0)
    plt.plot(np.arange(0,num_steps,1),avg_ep,label=tag)
    std_error1 = 1.96*np.std(episode_count)/np.sqrt(10)
    plt.fill_between(np.arange(0,num_steps,1),avg_ep - std_error1,avg_ep + std_error1,alpha=0.4)

    plt.legend()
    plt.xlabel("steps")
    plt.ylabel("episodes")
    plt.show()

def q4():
    env = WindyGridWorldEnv(king_move=False, stochastic=False)
    plot(env, num_steps=8000)

def q5():
    env = WindyGridWorldEnv(king_move=False, stochastic=False)
    Q = get_Q_val(env, 10000, 1, 0.1, 0.5)

    eps_1 = rollout(1, Q, env)
    eps_15 = rollout(15, Q, env)
    eps_20 = rollout(20, Q, env)
    eps_500 = rollout(500, Q, env)
    eps = eps_500

    _, lt_mc = monte_carlo_prediction(eps, 1)
    _, lt_td0 = td_prediction(env=env, gamma=1, episodes=eps, n=1)
    _, lt_td4 = td_prediction(env=env, gamma=1, episodes=eps, n=5)

    # plt.hist(lt_mc, bins=20, alpha=1, label=f"mc_{len(eps)}_eps")
    plt.hist(lt_td0, bins=20, alpha=1, label=f"td0_{len(eps)}_eps")
    # plt.hist(lt_td4, bins=20, alpha=1, label=f"4_step_td_{len(eps)}_eps")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # q4()
    q5()
