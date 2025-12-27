import gym  
import numpy as np  

env = gym.make("FrozenLake-v1", is_slippery=False)  
nS = env.observation_space.n
nA = env.action_space.n


def random_policy():
    return np.ones((nS, nA)) / nA 


def policy_evaluation(policy, gamma=0.99, theta=1e-6): 
    
    V = np.zeros(nS)

    while True: 
       delta = 0  
       for s in range(nS):  
           v = 0 
           for a in range(nA):  
               for prob, s_next, r, done in env.P[s][a]:  
                   v += policy[s, a] * prob * (r + gamma * V[s_next])
           delta = max(delta, abs(v - V[s]))
           V[s] = v  
       if delta < theta:
            break
    return V

def policy_improvement(V, gamma=0.99):
    policy = np.zeros((nS, nA))

    for s in range(nS):
        q_values = np.zeros(nA)
        for a in range(nA):
            for prob, s_next, r, done in env.P[s][a]:
                q_values[a] += prob * (r + gamma * V[s_next])
        best_a = np.argmax(q_values)
        policy[s, best_a] = 1.0
    return policy



def policy_iteration(gamma = 0.99):

      policy = random_policy()

      while True:  
          
          V = policy_evaluation(policy)
          new_policy = policy_improvement(V, gamma) 
          
          if np.all(policy == new_policy):
            break
          policy = new_policy
      
      return policy, V 

print(policy_iteration()) 




