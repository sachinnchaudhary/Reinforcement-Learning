import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

class Policynet(nn.Module):  
     
     def __init__(self, obs_dim, act_dim):  
          
        super().__init__()
        
        self.net = nn.Sequential(
        nn.Linear(obs_dim, 128),
        nn.ReLU(), 
        nn.Linear(128, act_dim), 
        nn.Softmax(-1))
     
     def forward(self, x):  
         
         return self.net(x)  
    
class Valuenet(nn.Module):  

      def __init__(self, obs_dim):  
          super().__init__()
          self.net = nn.Sequential(  
              nn.Linear(obs_dim, 128),
              nn.ReLU(),
              nn.Linear(128, 1),
          ) 
      
      def forward(self,x):  
          return self.net(x).squeeze(-1)


class ReinforceWithBaseline:  

    def __init__(self, env, lr_policy= 1e-3, lr_value= 1e-3, gamma =0.99):  
         super().__init__() 
         self.env= env 
         self.gamma = gamma  

         obs_dim = env.observation_space.shape[0]
         act_dim = env.action_space.n  

         self.policy = Policynet(obs_dim, act_dim) 
         self.value = Valuenet(obs_dim)
         self.policy_opt = optim.Adam(self.policy.parameters(),lr= lr_policy)
         self.value_opt = optim.Adam(self.value.parameters(),lr= lr_value)


    def select_action(self, state):  

        state = torch.tensor(state, dtype= torch.float32) 
        probs= self.policy(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        return action.item(), dist.log_prob(action)
    
    def train(self, episodes): 

        for episode in range(episodes):  
            states = []
            log_probs = []
            rewards = [] 

            state, _ = self.env.reset()
            done = False 

            while not done:  
                action, log_prob = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated  
                
                states.append(state)
                log_probs.append(log_prob)
                rewards.append(reward)

                state = next_state 

            returns = []
            G = 0

            for r in reversed(rewards):  

                G = r + self.gamma * G 
                returns.insert(0, G) 

            returns = torch.tensor(returns, dtype=torch.float32)
            states = torch.tensor(states, dtype= torch.float32)

            values = self.value(states)
            advantages = returns - values.detach()

            policy_loss = -(log_prob * advantages).sum()
            value_loss = ((values - returns) ** 2).mean() 
           
            self.policy_opt.zero_grad()
            policy_loss.backward()
            self.policy_opt.step()

          
            self.value_opt.zero_grad()
            value_loss.backward()
            self.value_opt.step()

            if episode % 50 == 0:
                print(f"Episode {episode}, return = {sum(rewards)}")
                
             
            

env = gym.make("CartPole-v1")
agent = ReinforceWithBaseline(env)

print(agent.train(episodes=1000)) 
