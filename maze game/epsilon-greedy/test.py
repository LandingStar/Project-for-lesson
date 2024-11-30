import numpy as np
import args
import classes
import sys
import threading
import random
max_column=0
max_row=0
sys.stdout=open("out.txt","w")
def import_mazz(path:str)->list:
    global max_column,max_row
    file=open(path,"r")
    n,m=tuple(map(lambda x:int(x),file.readline()[:-1].split(",")[:2]))
    max_row,max_column=n,m
    blocks=[classes.block([]) for i in range(m)for j in range(n)]
    value=[]
    def give_actions(pos:int)->list:
        r,c=[pos//m,pos%m]
        a=[-1,0,1]
        ret=[classes.action(np.array([(r+i)%n*m+(c+j)%m,]),np.array([1,]))for i in a for j in a]
        cnt=0
        for i in a:
            for j in a:
                ret[cnt].action_reward=value[(r+i)%n*m+(c+j)%m]
                ret[cnt].action_value=value[(r+i)%n*m+(c+j)%m]
                ret[cnt].visit_cnt=1
                cnt+=1
        return ret
    for i in range(n):
        lin=file.readline()[:-1].split("\t")
        value+=tuple(map(lambda x:np.float64(x),lin))
    for i in range(n*m):
        blocks[i].actions=give_actions(i)
    classes.space=classes.set(blocks,n,m)
    classes.space.maxlen=[n,m]
    return classes.space
classes.space=import_mazz("C:\\Users\\16329\\Source\\Repos\\LandingStar\\CST-Project\\maze game\\epsilon-greedy\\mazz.txt")
for cnt_round in range(1500):
    epsilon_policy=classes.policy(args.epsilon_greedy(100/(100+cnt_round)))
    init_state=random.randint(0,len(classes.space)-1)
    eps=classes.episode(init_state,epsilon_policy,args.episode_lenth)
    sample_return=0
    for s,a in eps.track[::-1]:
        sample_return=classes.space[s].actions[a].action_reward+args.gamma*sample_return
        classes.space[s].actions[a].update(sample_return)
while 0:
    init_position=input("beginning state").split(" ")[:2]
    init_state=classes.space[init_position[0]*max_column+init_position[1]]
    eps=classes.episode(args.epsilon_greedy(0),10)
    for i,j in eps.track:
        1
state_value=tuple(map(lambda x:max(tuple(map(lambda y:y.action_value,x.actions))),classes.space.blocks))
from sys import stdout
for i in range(len(classes.space)):
    if not i%max_column:
        stdout.write("\n")
    stdout.write(f"{state_value[i]:5.3f}\t")
stdout.write("\n")