import gym  
import numpy as np 
from collections import defaultdict 

env = gym.make("FrozenLake-v1", is_slippery=False)
n_states = env.observation_space.n
n_actions = env.action_space.n

gamma = 0.99  

def random_policy(state):
    return np.random.choice(n_actions) 



def mc_state_value(env, policy, num_episodes=5000): 

    V = np.zeros(n_states)
    returns = defaultdict(list) 

    for episode in range(num_episodes):  
        state, _ = env.reset()
        episode_states = []
        episode_rewards = []

        done = False
        while not done:  

            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_states.append(state)
            episode_rewards.append(reward)
            state = next_state
            done = terminated or truncated
        
        G = 0  
        visited = set()
        for t in reversed(range(len(episode_states))): 
            s = episode_states[t]
            G = gamma * G + episode_rewards[t] 
            if s not in visited:  # first-visit
                returns[s].append(G)
                V[s] = np.mean(returns[s])
                visited.add(s)

    return V 

V_mc = mc_state_value(env, random_policy).reshape(4,4)

def greedy_policy_from_Q(Q):
    return np.argmax(Q, axis=1) 

print(V_mc) 
print(greedy_policy_from_Q(V_mc))
