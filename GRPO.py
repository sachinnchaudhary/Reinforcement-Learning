import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim   

env = gym.make("FrozenLake-v1", is_slippery = False) 

class Policy(nn.Module):  

     def __init__(self, obs_dim, act_dim):  
          
        super().__init__() 
        self.logits = nn.Embedding(obs_dim, act_dim)

     def forward(self, state):  
         logits = self.logits(torch.tensor(state))
         return torch.softmax(logits, dim=-1)
     
    
class GRPOagent:  

    def __init__(self, env, group_size=8, clip_eps=0.2, beta_kl= 0.01, lr=1e-2):
      
       self.env = env 
       self.group_size = group_size 
       self.clip_eps = clip_eps  
       self.beta_kl = beta_kl 
       
       self.n_states = env.observation_space.n
       self.n_actions = env.action_space.n  

       self.policy = Policy(self.n_states, self.n_actions)
       self.reference_policy = Policy(self.n_states, self.n_actions)

       self.optimizer = optim.Adam(self.policy.parameters(), lr=lr) 
    
    def sample_episode(self, policy): 

        states, actions, log_probs = [], [], []
        state, _ = self.env.reset()
        done = False 

        while not done:  

            s = torch.tensor(state) 
            probs = policy(s)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            
            next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            done = terminated or truncated 

            states.append(state)
            actions.append(action)
            log_probs.append(dist.log_prob(action).detach())

            state = next_state
             
        return states, actions, log_probs, reward 
    

    def update(self):

        batch = []

        for _ in range(self.group_size):  
            traj = self.sample_episode(self.policy)
            batch.append(traj)

        rewards = torch.tensor([traj[3] for traj in batch], dtype=torch.float32)

        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)  

        policy_loss = 0 
        kl_loss = 0 
        count = 0  

        for (state, action, old_log_prob, _), adv in zip(batch, advantages):  

            for s, a, old_lp in zip(state, action, old_log_prob):  
                probs = self.policy(s)
                dist = torch.distributions.Categorical(probs)
                new_lp = dist.log_prob(a)

                ratio = torch.exp(new_lp - old_lp)
                clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                policy_loss += -torch.min(ratio * adv, clipped *adv)
                 
                with torch.no_grad():  
                     ref_probs = self.reference_policy(s) 
                
                kl_loss += torch.sum(ref_probs * (torch.log(ref_probs) - torch.log(probs)))

                count += 1  

        loss = policy_loss / count + self.beta_kl * kl_loss / count 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.reference_policy.load_state_dict(self.policy.state_dict())
        return rewards.mean().item()
    

    def train(self, episodes = 500):  

        for episode in range(episodes): 
            avg_reward = self.update()
            if episode % 20 == 0:  
                print(f"episode {episode}, avg_reward = {avg_reward}")

if __name__ == "__main__":
 agent = GRPOagent(env)
 agent.train()

 


        



