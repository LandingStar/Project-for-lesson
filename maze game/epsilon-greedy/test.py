import numpy as np
import args
from classes import *
import threading
def import_mazz(path:str)->list:
    
    file=open(path,"r")
    n,m=file.readline()[-1].split(",")[:2]
    blocks=[block([]) for i in range(m)for j in range(n)]
    value=[]
    def give_actions(pos:int)->tuple:
        r,c=[pos//m,pos%m]
        a=[-1,0,1]
        pos=[ (r+i)%n*m+(c+j)%m for i in a for j in a]
        reward=[value[(r+i)%n*m+(c+j)%m]for i in a for j in a]
        ret= action([pos,],np.array([1,]))
        ret=[action([blocks[(r+i)%n*m+(c+j)%m],],[1,])for i in a for j in a]
        cnt=0
        for i in a:
            for j in a:
                ret[cnt].action_reward=value[(r+i)%n*m+(c+j)%m]
                cnt+=1
        return ret
    for i in range(n):
        blocks.append(map(lambda x:np.float64(x),file.readline()[-1].split(" ")))
    for i in range(n*m):
        blocks[i].actions=[give_actions(i)]
    return blocks
