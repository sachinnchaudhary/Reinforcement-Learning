import gym  
import numpy as np   


env = gym.make("FrozenLake-v1", is_slippery=False)  
nS = env.observation_space.n
nA = env.action_space.n


def random_policy():
    return np.ones((nS, nA)) / nA 

 
def value_iteration(gamma = 0.99, theta = 1e-6):  

    V = np.zeros(nS)  
    policy = np.zeros(nS, dtype= int)

    while True:  
        delta = 0 
        for s in range(nS):  
            q_values = np.zeros(nA) 
            for a in range(nA):  
                 for prob, s_next, reward, done in env.P[s][a]:  
                      q_values[a] += prob * (reward + gamma * V[s_next]) 
            v_new = np.max(q_values)
            policy[s] = np.argmax(q_values)
            delta = max(delta, abs(v_new - V[s])) 
            V[s] = v_new 
        if delta < theta: 
            break  
    return V, q_values

print(value_iteration())
    