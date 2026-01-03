import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim   

env = gym.make("CartPole-v1")   

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU()
        )
        self.policy_head = nn.Sequential(
            nn.Linear(128, act_dim),
            nn.Softmax(dim=-1)
        )
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        policy = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return policy, value  
    

class A2Cgae:   

    def __init__(self, env, gamma = 0.99, lam=0.95, lr=1e-3):
           
           self.env = env  
           self.gamma = gamma  
           self.lam = lam   

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
    

    def train(self, episodes = 1000):  
         
         for episode in range(episodes):   
            states = []
            log_probs = []
            rewards = []
            values = [] 
         
            state, _  = self.env.reset()
            done = False 
            total_reward =  0   

            while not done:  
                 action, log_prob, value = self.select_action(state)  
                 next_state, reward, terminated, truncated, _ = self.env.step(action)
                 done = terminated or truncated  
                 
                 states.append(state)
                 log_probs.append(log_prob)
                 rewards.append(reward)
                 values.append(value)

                 state = next_state 
                 total_reward += reward 

            values = torch.stack(values)

            with torch.no_grad():  
                 next_value = torch.tensor(0.0)

            deltas = []

            for t in range(len(rewards)):  
                 delta = rewards[t] + self.gamma * (next_value if t == len(rewards)-1 else values[t+1]) - values[t]
                 deltas.append(delta) 
            
            advantages = [] 
            gae = 0  
            for delta in reversed(deltas):
                gae = delta + self.gamma * self.lam * gae
                advantages.insert(0, gae)
            advantages = torch.stack(advantages).detach()
            returns = advantages + values.detach()

            policy_loss = -(torch.stack(log_probs) * advantages).sum()
            value_loss = (values - returns).pow(2).mean()

            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if episode % 50 == 0:
                print(f"Episode {episode}, return = {total_reward}")
                
if __name__ == "__main__":  
 agent = A2Cgae(env)
 agent.train(episodes=1000)


                
            