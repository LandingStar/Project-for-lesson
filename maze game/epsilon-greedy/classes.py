import numpy as np
import random

from typing import Callable

class block:
	def __init__(self,actions:list):
		self.state_value=0
		self.actions=actions
		self.action_values=np.array([0]*len(actions))

class action:
	def __init__(self,next_state:list,next_state_p:np.ndarray):
		self.action_value=0
		self.action_reward=0
		self.visit_cnt=0 #times that be visited
		self.next_state=next_state
		self.next_state_p=next_state_p
		#self.next_state_p_expected=np.array([1/len(next_state)]*len(next_state)) #the expected position after some action used for questions that the next state is not settled after a certain action
	def __iter__(self):
		return self
	def __next__(self)->block: #giving next state by taking this action
		total=np.sum(self.next_state_p)
		x=random.random()*total
		for i in range(len(self.next_state_p)):
			x-=self.next_state_p[i]
			if x<=0:
				return self.next_state[i]
		return self.next_state[0]
	def update(self,ret,a:Callable[[int,],float]=lambda x:1/x): # 'a' is the conerge coiffiecient in RM
		self.action_value=self.action_value-a(self.visit_cnt)*(self.action_value-ret)
		self.visit_cnt+=1





class policy:
	def __init__(self,principle:Callable[[np.ndarray,],np.ndarray]):
		policy.principle=principle
	def __call__(self,state:block)->np.ndarray: #call the instance then get probability of chocing actions
		return self.principle(state.action_values)
	def choice(self,state:block)->action:       #return the action the policy choose
		probability=self.__call__(state)
		total=probability.sum()
		x=total*random.random()
		for i in range(probability):
			x-=probability[i]
			if x<=0:
				return self.next_state[i]
		return self.next_state[0]

class episode:
	def __init__(self,state:block,policy:policy,step:int):# use a initial state to create an episode with lenth of 'step' by the policy
		self.track=[]
		self.now_state=state
		self.policy=policy
		for i in range(step):
			a=policy.choice(self.now_state)
			#a=self.act(policy)
			self.track.append((self.now_state,a))
			self.now_state=next(a)
	def __iter__(self)->list:
		return self.track
	def act(self,policy:policy)->action:				# return the action the policy choose  ,same as choice in class policy
		return policy.choice()
