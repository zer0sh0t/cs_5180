from env import Action
import numpy as np
from scipy.special import softmax
from env import *

def argmax(arr):
    max_value = float('-inf')
    max_indices = []

    for i, value in enumerate(arr):
        if value > max_value:
            max_value = value
            max_indices = [i]
        elif value == max_value:
            max_indices.append(i)

    return max_indices

def policy_evaluation(policy, gamma, theta, transitions, rows, cols, display_it):
    V = np.zeros((rows, cols))
    it = 0

    while True:
        delta = 0

        for x in range(cols):
            for y in range(rows):

                state = (x, y)
                old_v = V[y, x]
                new_v = 0

                for a in Action:
                    (next_x, next_y), reward = transitions(state, Action(a))
                    new_v += policy[y, x, a] * (reward + gamma * V[next_y, next_x])

                V[y, x] = new_v
                delta = max(delta, abs(old_v - V[y, x]))

        if display_it:
            print(it)

        it += 1

        if delta < theta:
            break

    return V

def value_iteration(gamma, theta, transitions, rows, cols):
    V = np.zeros((rows, cols))
    it = 0

    while True:
        delta = 0

        for x in range(cols):
            for y in range(rows):

                state = (x, y)
                old_v = V[y, x]
                action_values = []

                for a in Action:
                    (next_x, next_y), reward = transitions(state, Action(a))
                    action_values.append(reward + gamma * V[next_y, next_x])

                V[y, x] = max(action_values)
                delta = max(delta, abs(old_v - V[y, x]))

        print(it)
        it += 1

        if delta < theta:
            break

    policy = np.zeros((rows, cols), dtype=int)
    policy = policy.tolist()
    
    for x in range(cols):
        for y in range(rows):
            state = (x, y)
            action_values = []
            
            for a in Action:
                (next_x, next_y), reward = transitions(state, Action(a))
                action_values.append(reward + gamma * V[next_y, next_x])

            policy[y][x] = argmax(action_values)

    return V, policy

def policy_improvement(V, policy, gamma, theta, transitions, rows, cols):
    policy_stable = True

    for x in range(cols):
        for y in range(rows):

            state = (x, y)
            old_action = argmax(policy[y, x])
            action_values = []

            for a in Action:
                (next_x, next_y), reward = transitions(state, Action(a))
                action_values.append(reward + gamma * V[next_y, next_x])

            policy[y, x] = action_values

            if old_action != argmax(policy[y, x]):
                policy_stable = False
    
    return policy, policy_stable

def policy_iteration(gamma, theta, transitions, rows, cols):
    policy = softmax(np.random.rand(rows, cols, len(Action)))
    
    it = 0
    while True:
        V = policy_evaluation(policy, gamma, theta, transitions, rows, cols, False)
        policy, policy_stable = policy_improvement(V, policy, gamma, theta, transitions, rows, cols)
        policy = softmax(policy, -1)

        print(it)
        it += 1
        
        if policy_stable:
            break

    actual_policy = np.zeros((rows, cols), dtype=int)
    actual_policy = actual_policy.tolist()

    for x in range(cols):
        for y in range(rows):
            actual_policy[y][x] = argmax(policy[y, x])

    return V, actual_policy

def policy_evaluation_6(env, policy, gamma, theta, display_it):
    V = np.zeros((env.max_cars_end+1, env.max_cars_end+1))
    it = 0

    while True:
        delta = 0

        for A in range(env.max_cars_end+1):
            for B in range(env.max_cars_end+1):

                state = (A, B)
                old_v = V[A, B]
                new_v = 0

                for ia, action in enumerate(env.action_space):
                    expected_return = env.expected_return(V, state, action, gamma)
                    new_v += policy[A, B, ia] * expected_return

                V[A, B] = new_v
                delta = max(delta, abs(old_v - V[A, B]))

        if display_it:
            print(it)

        it += 1

        if delta < theta:
            break

    return V

def policy_improvement_6(env, V, policy, gamma, theta):
    policy_stable = True

    for A in range(env.max_cars_end+1):
        for B in range(env.max_cars_end+1):

            state = (A, B)
            old_action = argmax(policy[A, B])
            action_values = []

            for ia, action in enumerate(env.action_space):
                val = 0

                for next_state in env.state_space:
                    transitions = env.transitions(state, action)
                    val += transitions[next_state[0], next_state[1]] * (env.rewards(state, action) + gamma * V[next_state[0], next_state[1]])

                action_values.append(val)

            policy[A, B] = action_values

            if old_action != argmax(policy[A, B]):
                policy_stable = False
    
    return policy, policy_stable


def policy_iteration_6(env, gamma, theta):
    policy = softmax(np.random.rand(env.max_cars_end+1, env.max_cars_end+1, len(env.action_space)))

    it = 0
    while True:
        V = policy_evaluation_6(env, policy, gamma, theta, False)
        policy, policy_stable = policy_improvement_6(env, V, policy, gamma, theta)
        policy = softmax(policy, -1)

        print(it)
        it += 1

        if policy_stable:
            break

    actual_policy = np.zeros((env.max_cars_end+1, env.max_cars_end+1), dtype=int)
    actual_policy = actual_policy.tolist()

    for A in range(env.max_cars_end+1):
        for B in range(env.max_cars_end+1):
            actual_policy[A][B] = np.argmax(policy[A, B])

    return V, actual_policy
