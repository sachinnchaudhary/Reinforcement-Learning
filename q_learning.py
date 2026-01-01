import gymnasium as gym 
import numpy as np 

class QlearningAgent:   

     def __init__(self, env, alpha =0.1, gamma = 0.99, epsilon = 0.1):  
          
          self.env = env  
          self.alpha = alpha
          self.gamma = gamma 
          self.epsilon = epsilon  

          self.n_states = env.observation_space.n  
          self.n_action = env.action_space.n

          self.Q = np.zeros((self.n_states, self.n_action))

     def policy(self, state):
          if np.random.rand() > self.epsilon:  
             return self.env.action_space.sample() 
          return np.argmax(self.Q[state])
     
     def train(self, episode = 5000):   
          for _ in range(episode):  
               state, _ = self.env.reset()
               action = self.policy(state)
               done = False  

               while not done:  
                next_state, reward, terminated, truncated, _ = self.env.step(action)  
                done = terminated or truncated
                next_action = self.policy(next_state)

                target = reward + self.gamma * np.max(self.Q[next_state]) 
                self.Q[state, action] += self.alpha * (target - self.Q[state, action])

                state, action = next_state, next_action
                print(f"reward for state {state} is {reward}")

          return self.Q 

env = gym.make("FrozenLake-v1", is_slippery = False)
agent = QlearningAgent(env= env)

print(agent.train())
                 
           

