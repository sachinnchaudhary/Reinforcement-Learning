import gym  
import numpy as np  

env = gym.make("FrozenLake-v1", is_slippery=False)  
nS = env.observation_space.n
nA = env.action_space.n


def random_policy():
    return np.ones((nS, nA)) / nA 


def policy_evaluation(policy, gamma=0.9, theta=1e-6): 
    
    V = np.zeros(nS)

    while True: 
       delta = 0  
       for s in range(nS):  
           v = 0 
           for a in range(nA):  
               for prob, s_next, r, done in env.P[s][a]:  
                   v += policy[s, a] * prob * (r + gamma * V[s_next])
           delta = max(delta, abs(v - V[s]))
           print(delta)
           V[s] = v  
       if delta < theta:
            break
    return V

policy = random_policy()

Value_function = policy_evaluation(policy) 

print(Value_function)


