import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim  


class ActorCritic(nn.Module):  
     
     def __init__(self, obs_dim, act_dim):  
         super().__init__()
         self.shared = nn.Sequential(
              nn.Linear(obs_dim,128),
              nn.ReLU())
         
         self.policy_head =  nn.Sequential(  
                     nn.Linear(128, act_dim))

         self.value_head =  nn.Linear(128, 1)
          
     def forward(self, x):   
          
          x = self.shared(x)
          policy = self.policy_head(x)
          value = self.value_head(x).squeeze(-1)  

          return policy, value  
     

class A2Cagent:   
     
    def __init__(self, env, gamma=0.99, lr=1e-3):
        self.env = env
        self.gamma = gamma

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n

        self.model = ActorCritic(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        probs, value = self.model(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value 
        
    def train(self, episodes= 1000):

        for episode in range(episodes):  

             state, _ = self.env.reset() 
             done = False 
             total_reward = 0

             while not done:  
               action, log_prob, value = self.select_action(state)
               next_state, reward, terminated, truncated, _ = self.env.step(action)
               done = terminated or truncated
               total_reward += reward

               next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
               with torch.no_grad():
                    _, next_value = self.model(next_state_tensor)
                
               td_target = reward + self.gamma * next_value * (1 - int(done))
               advantage = td_target - value

               policy_loss = -log_prob * advantage.detach()
               value_loss = advantage.pow(2)

               loss = policy_loss + value_loss

               self.optimizer.zero_grad()
               loss.backward()
               self.optimizer.step()

               state = next_state

               if episode % 50 == 0:
                 print(f"Episode {episode}, return = {total_reward}")

if __name__ == "__main__": 
 env = gym.make("CartPole-v1")
 agent = A2Cagent(env)

 agent.train(episodes= 1000)

 


           
        
          
 