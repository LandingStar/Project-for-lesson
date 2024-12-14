import numpy as np
import random
#from numba import jit
from typing import Callable
global space
space=[]
class block:
	def __init__(self,actions:list):
		self.position=0
		self.state_value=0
		self.state_reward=0
		self.actions=actions
		self.action_values=np.array([0]*len(actions))
class set():
	def __init__(self,blocks:list,*maxlen):
		self.blocks=list(blocks)
		self.maxlen=list(maxlen)
		prod=1
		for i in maxlen:
			prod*=i
		self.length=prod
		cnt=0
		for i in self.blocks:
			i.postion=cnt
			cnt+=1
	def __len__(self):
		return self.length
	def __iter__(self):
		return self.blocks.__iter__()
	def __getitem__(self,index):
		return self.blocks[index]
	def getitem(self,*keys):
		position=0
		base=1
		for i in range(len(self.maxlen)-1,-1,-1):
			position+=keys[i]*base
			base*=self.maxlen[i]
		return self.blocks[position]
	def get_index(self,*keys):
		position=0
		base=1
		for i in range(len(self.maxlen)-1,-1,-1):
			position+=keys[i]*base
			base*=self.maxlen[i]
		return position
class action:
	def __init__(self,next_state:list,next_state_p:np.ndarray):
		self.action_value=0
		self.action_reward=0
		self.visit_cnt=0 #times that be visited
		self.next_state=next_state
		self.next_state_p=next_state_p
		self.direction=[]
		#self.next_state_p_expected=np.array([1/len(next_state)]*len(next_state)) #the expected position after some action used for questions that the next state is not settled after a certain action
	def __iter__(self):
		return self
	def __next__(self)->int: #giving next state by taking this action
		total=np.sum(self.next_state_p)
		x=random.random()*total
		for i in range(len(self.next_state_p)):
			x-=self.next_state_p[i]
			if x<=0:
				return self.next_state[i]
		return self.next_state[0]
	def update(self,ret,a:Callable[[int,],float]=lambda x:1/x): # 'a' is the conerge coiffiecient in RM
		self.visit_cnt+=1
		self.action_value=self.action_value-a(self.visit_cnt)*(self.action_value-ret)
		





class policy:
	def __init__(self,principle:Callable[[np.ndarray,],np.ndarray]):
		self.principle=principle
	def __call__(self,state:block)->np.ndarray: #call the instance then get probability of chocing actions
		values=list(map(lambda x:x.action_value,state.actions))
		return self.principle(values)
	def choice(self,state:block)->action:       #return the action the policy choose
		probability=self.__call__(state)
		total=probability.sum()
		x=total*random.random()
		for i in range(len(probability)):
			x-=probability[i]
			if x<=0:
				return i
		return 0

class episode:
	def __init__(self,state:block,policy:policy,step:int):# use a initial state to create an episode with lenth of 'step' by the policy
		self.track=[]
		self.now_state=state
		self.policy=policy
		if step<0:
			while space[self.now_state].state_reward<=0:
				a=policy.choice(space[self.now_state])
				#a=self.act(policy)
				self.track.append((self.now_state,a))
				self.now_state=next(space[self.now_state].actions[a])
			return
		for i in range(step):
			a=policy.choice(space[self.now_state])
			#a=self.act(policy)
			self.track.append((self.now_state,a))
			self.now_state=next(space[self.now_state].actions[a])
			if space[self.now_state].state_reward>0:
				return
	def __iter__(self)->list:
		return self.track
	def act(self,policy:policy)->action:				# return the action the policy choose  ,same as choice in class policy
		return policy.choice()
