import numpy as np
from typing import Callable
episode_lenth=250   #the lenth of 
gamma=0.9
convergence_critirion=0.01
reward_common_road=0
reward_forbidden_area=-1
reward_target=10
def epsilon_greedy(epsilon:float)->Callable[[np.ndarray,],np.ndarray]: #give a function giving the probability choosing actions under epsilon-greedy policy.Used in <class policy>.__init__
    def the_policy(action_values:np.ndarray):
        cnt=len(action_values)
        p=np.array([epsilon/cnt]*cnt)
        p[np.argmax(action_values)]+=1-epsilon
        return p
    return the_policy