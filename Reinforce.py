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
    

class ReinforceAgent:  

    def __init__(self, env, lr= 1e-3, gamma =0.99):  
         super().__init__()
         self.env= env 
         self.gamma = gamma  

         obs_dim = env.observation_space.shape[0]
         act_dim = env.action_space.n  

         self.policy = Policynet(obs_dim, act_dim) 
         self.optimizer = optim.Adam(self.policy.parameters(),lr= lr)

    def select_action(self, state):  

        state = torch.tensor(state, dtype= torch.float32) 
        probs= self.policy(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        return action.item(), dist.log_prob(action)
    
    def train(self, episodes): 

        for episode in range(episodes):  
            log_probs = []
            rewards = []

            state, _ = self.env.reset()
            done = False 

            while not done:  
                action, log_prob = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated  

                log_probs.append(log_prob)
                rewards.append(reward)

                state = next_state 

            returns = []
            G = 0

            for r in reversed(rewards):  

                G = r + self.gamma * G 
                returns.insert(0, G) 

            returns = torch.tensor(returns, dtype=torch.float32)

            loss = 0  
            for log_prob, gt in zip(log_probs, returns):
                loss += -log_prob * gt 
                
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if episode % 50 == 0:  
                print(f"Episode {episode}, return = {sum(rewards)}")
             
            

env = gym.make("CartPole-v1")
agent = ReinforceAgent(env)

print(agent.train(episodes=1000)) 

