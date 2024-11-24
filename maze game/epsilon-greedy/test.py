import numpy as np
import args
from classes import *
import threading
max_column=0
max_row=0
def import_mazz(path:str)->list:
    global max_column,max_row
    file=open(path,"r")
    n,m=tuple(map(lambda x:int(x),file.readline()[:-1].split(",")[:2]))
    max_row,max_column=n,m
    blocks=[block([]) for i in range(m)for j in range(n)]
    value=[]
    def give_actions(pos:int)->list:
        r,c=[pos//m,pos%m]
        a=[-1,0,1]
        ret=[action(np.array([blocks[(r+i)%n*m+(c+j)%m],]),np.array([1,]))for i in a for j in a]
        cnt=0
        for i in a:
            for j in a:
                ret[cnt].action_reward=value[(r+i)%n*m+(c+j)%m]
                ret[cnt].action_value=value[(r+i)%n*m+(c+j)%m]
                ret[cnt].visit_cnt=1
                cnt+=1
        return ret
    for i in range(n):
        lin=file.readline()[:-1].split(" ")
        value+=tuple(map(lambda x:np.float64(x),lin))
    for i in range(n*m):
        blocks[i].actions=give_actions(i)
    return blocks
blocks=import_mazz("C:\\Users\\16329\\Source\\Repos\\LandingStar\\CST-Project\\maze game\\epsilon-greedy\\mazz.txt")
for cnt_round in range(1000):
    epsilon_policy=policy(args.epsilon_greedy(100/(150+cnt_round**2)))
    init_state=blocks[random.randint(0,len(blocks)-1)]
    eps=episode(init_state,epsilon_policy,args.episode_lenth)
    sample_return=0
    for s,a in eps.track[::-1]:
        sample_return=a.action_reward+args.gamma*sample_return
        a.update(sample_return)
while 0:
    init_position=input("beginning state").split(" ")[:2]
    init_state=blocks[init_position[0]*max_column+init_position[1]]
    eps=episode(args.epsilon_greedy(0),10)
    for i,j in eps.track:
        1
state_value=tuple(map(lambda x:max(tuple(map(lambda y:y.action_value,x.actions))),blocks))
from sys import stdout
for i in range(len(blocks)):
    if not i%max_column:
        stdout.write("\n")
    stdout.write(str(state_value[i])+"\t")
stdout.write("\n")