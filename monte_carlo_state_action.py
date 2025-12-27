import gym  
import numpy as np 
from collections import defaultdict 

env = gym.make("FrozenLake-v1", is_slippery=False)
n_states = env.observation_space.n
n_actions = env.action_space.n

gamma = 0.99  

def random_policy(state):
    return np.random.choice(n_actions) 
 
def mc_state_action(env, policy, episodes = 5000):  
     
    Q = np.zeros((n_states, n_actions))
    returns = defaultdict(list) 

    for episode in range(episodes):
        state, _ = env.reset()
        episode_sa = []
        episode_rewards = [] 

        done = False 
        while not done:  
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_sa.append((state, action))
            episode_rewards.append(reward)
            state = next_state
            done = terminated or truncated 
        
        G = 0  
        visited = set()
        for t in reversed(range(len(episode_sa))):
            sa = episode_sa[t]
            G = gamma * G + episode_rewards[t]

            if sa not in visited:
                returns[sa].append(G)
                s, a = sa
                Q[s, a] = np.mean(returns[sa])
                visited.add(sa) 

    return Q

Q_mc = mc_state_action(env, random_policy)

def greedy_policy_from_Q(Q):
    return np.argmax(Q, axis=1)

print(Q_mc) 
print(greedy_policy_from_Q(Q_mc))
