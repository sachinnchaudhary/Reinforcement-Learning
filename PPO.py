import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim    


class ActorCritic(nn.Module):

    def __init__(self, obs_dim, act_dim):   

         super().__init__()  
         self.shared = nn.Sequential(
              nn.Linear(obs_dim, 128), 
              nn.ReLU()
         )

         self.policy_head = nn.Sequential(nn.Linear(128, act_dim),
                                          nn.Softmax(-1))
         self.value_head = nn.Linear(128,1)  

    def forward(self, x):  
        
        x = self.shared(x)  
        policy = self.policy_head(x)  
        value = self.value_head(x).squeeze(-1)  

        return policy, value  
     

class PPOagent:  

     def __init__(self, env, gamma =0.99, lam= 0.95, clip_eps=0.2, lr=3e-4, epochs = 4, batch_size=2048):

        super().__init__()  
        self.env = env 
        self.gamma= gamma 
        self.lam = lam  
        self.clip_eps = clip_eps 
        self.lr = lr  
        self.epochs = epochs  
        self.batch_size = batch_size 
    
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n

        self.model = ActorCritic(obs_dim, act_dim) 
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr) 
     
     def select_action(self, state):  
         
        state = torch.tensor(state, dtype= torch.float32)  
        probs, value= self.model(state) 
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value 
     
     def collect_rollout(self):  
    
           states, actions, rewards = [], [], []
           log_probs, values, dones = [], [], []

           state , _ = self.env.reset()
           total_steps = 0

           while total_steps < self.batch_size:  
               action, log_prob, value = self.select_action(state)
               next_state, reward, terminated, truncated, _ = self.env.step(action)
               done = terminated or truncated 
            
               states.append(state)
               actions.append(action)
               rewards.append(reward)
               log_probs.append(log_prob)
               values.append(value)
               dones.append(done)

               state = next_state if not done else self.env.reset()[0] 
               total_steps +=1  
              
           return (torch.tensor(states, dtype=torch.float32),
                      torch.tensor(actions),
                      torch.tensor(rewards, dtype=torch.float32), 
                      torch.stack(log_probs),
                      torch.stack(values), 
                      torch.tensor(dones, dtype=torch.float32)) 
           
           
     def GAE(self, rewards, values, dones):

          advantages = [] 
          gae = 0  

          values =torch.cat([values, torch.tensor([0.0])])

          for t in reversed(range(len(rewards))): 
              delta =  (
                  rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
              )
              gae = delta + self.gamma * self.lam * (1 -dones[t]) * gae 
              advantages.insert(0, gae)

          return torch.tensor(advantages, dtype=torch.float32)
     
     def update(self):  
         
         states, actions, rewards, old_log_probs, values, dones = self.collect_rollout()
         advantages = self.GAE(rewards, values, dones)
         
         old_log_probs = old_log_probs.detach() 
         
         returns = advantages + values.detach()
         
         advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
         
         for _ in range(self.epochs):  
             probs, new_values = self.model(states)
             dist = torch.distributions.Categorical(probs)

             new_log_probs = dist.log_prob(actions) 
             ratio = torch.exp(new_log_probs - old_log_probs)

             clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 +self.clip_eps)

             policy_loss = -torch.mean(torch.min(ratio * advantages, clipped_ratio * advantages))
             value_loss = torch.mean((returns - new_values) ** 2) 
              
             entropy = dist.entropy().mean()   
             loss = policy_loss + 0.5 * value_loss - 0.01  * entropy

             self.optimizer.zero_grad()
             loss.backward()
             self.optimizer.step()

     def train(self, iterations = 200): 
         for i in range(iterations):  
             self.update() 
             if i % 10 == 0:  
                 print(f"iteration {i}: reward collected {sum(self.collect_rollout()[2])}")


agent = PPOagent(gym.make("CartPole-v1"))
agent.train(iterations=200)





         

          


           
