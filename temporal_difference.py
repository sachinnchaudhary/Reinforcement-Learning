import gymnasium as gym  
import numpy as np  


import gymnasium as gym
import numpy as np

class TDAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        
        self.n_states = env.observation_space.n
        self.V = np.zeros(self.n_states)
    
    def policy(self, state):
        # Fixed random policy (π)
        return self.env.action_space.sample()
    
    def td_update(self, state, reward, next_state, done):
        target = reward if done else reward + self.gamma * self.V[next_state]
        td_error = target - self.V[state]
        self.V[state] += self.alpha * td_error
        return td_error
    
    def train(self, episodes=1000):
        for _ in range(episodes):
            state, _ = self.env.reset()
            done = False
            
            while not done:
                action = self.policy(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                self.td_update(state, reward, next_state, done)
                state = next_state 

import gymnasium as gym
import numpy as np

class TDAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        
        self.n_states = env.observation_space.n
        self.V = np.zeros(self.n_states)
    
    def policy(self, state):
        # Fixed random policy (π)
        return self.env.action_space.sample()
    
    def td_update(self, state, reward, next_state, done):
        target = reward if done else reward + self.gamma * self.V[next_state]
        td_error = target - self.V[state]
        self.V[state] += self.alpha * td_error
        return td_error
    
    def train(self, episodes=1000):
        for _ in range(episodes):
            state, _ = self.env.reset()
            done = False
            
            while not done:
                action = self.policy(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                self.td_update(state, reward, next_state, done)
                state = next_state

env = gym.make("FrozenLake-v1", is_slippery=False)
agent = TDAgent(env, alpha=0.1, gamma=0.99)

agent.train(episodes=5000)

print("Learned V(s):")
print(agent.V.reshape(4, 4))