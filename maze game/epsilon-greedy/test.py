import numpy as np
import args
import classes
import sys
import threading
import random
import matplotlib.pyplot as plt
import math
import multiprocessing
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
        #a=((0,0),(-1,0),(1,0),(0,-1),(0,1))
        ret=[classes.action(np.array([(r+i)%n*m+(c+j)%m,]),np.array([1,]))for i in a for j in a]
        #ret=[classes.action(np.array([(r+i)%n*m+(c+j)%m,]),np.array([1,]))for (i,j) in a]
        cnt=0
        for i in a:
            for j in a:
                ret[cnt].action_reward=value[(r+i)%n*m+(c+j)%m]
                ret[cnt].action_value=value[(r+i)%n*m+(c+j)%m]
                ret[cnt].visit_cnt=1
                ret[cnt].direction = (i, j)
                cnt+=1
        #for i,j in a:
        #    if value[(r+i)%n*m+(c+j)%m]<0:
        #        ret[cnt].next_state=[(r)%n*m+(c)%m,]
        #    ret[cnt].action_reward=value[(r+i)%n*m+(c+j)%m]
        #    ret[cnt].action_value=value[(r+i)%n*m+(c+j)%m]
        #    ret[cnt].visit_cnt=1
        #    cnt+=1
        return ret
    for i in range(n):
        lin=file.readline()[:-1].split("\t")
        value+=tuple(map(lambda x:np.float64(x),lin))
    for i in range(n*m):
        blocks[i].actions=give_actions(i)
        blocks[i].state_reward=value[i]
    classes.space=classes.set(blocks,n,m)
    classes.space.maxlen=[n,m]
    return classes.space
classes.space=import_mazz("mazz.txt")

init_state=random.randint(0,len(classes.space)-1)
sign=1
#process_pool=multiprocessing.Pool()
for cnt_round in range(args.run_round):
    epsilon_policy=classes.policy(args.epsilon_greedy(1))

    #epsilon_policy=classes.policy(args.random_choice(5))

    init_state=random.randint(0,len(classes.space)-1)
    eps=classes.episode(init_state,epsilon_policy,args.episode_lenth)
    #sample_return=classes.space[eps.track[-1][0]].actions[eps.track[-1][1]].action_value
    for ind in range(len(eps.track)-2,0,-1):
        s,a=eps.track[ind]

        #sample_return=classes.space[s].actions[a].action_reward+args.gamma*sample_return
        #classes.space[s].actions[a].update(sample_return,lambda x:0.001)
        classes.space[s].actions[a].update(classes.space[s].actions[a].action_reward+args.gamma*max(list(map(lambda x:x.action_value,classes.space[eps.track[ind+1][0]].actions))))
while 0:
    init_position=input("beginning state").split(" ")[:2]
    init_state=classes.space[init_position[0]*max_column+init_position[1]]
    eps=classes.episode(args.epsilon_greedy(0),10)
    for i,j in eps.track:
        1




















state_value=tuple(map(lambda x:max(tuple(map(lambda y:y.action_value,x.actions))),classes.space.blocks))
fig1 = plt.figure(num=1, figsize=(max_column,max_row))
axes1 = fig1.add_subplot(1,1,1)
for i in range(len(classes.space)):
    y=i//max_column
    x=i%max_column
    if not i%max_column:
        sys.stdout.write("\n")
    sys.stdout.write(f"{state_value[i]:5.3f}\t")

    square = plt.Rectangle(xy=(1/max_column*x, 1/max_row*y), width=1/max_column, height=1/max_row, alpha=0.8, angle=0.0,color=(max(0,-math.tanh(0.2*(state_value[i]+classes.space[i].state_reward))),0,max(0,math.tanh(0.2*(state_value[i]+classes.space[i].state_reward))),0))

    axes1.add_patch(square)




plt.savefig("out.png")
plt.close()
fig1 = plt.figure(num=1, figsize=(max_column,max_row))
axes1 = fig1.add_subplot(1,1,1)
for i in range(len(classes.space)):
    y=i//max_column
    x=i%max_column
    #if not i%max_column:
    #    stdout.write("\n")
    #stdout.write(f"{state_value[i]:5.3f}\t")

    square = plt.Rectangle(xy=(1/max_column*x, 1/max_row*y), width=1/max_column, height=1/max_row, alpha=0.8, angle=0.0,color=(max(0,-math.tanh(classes.space[i].state_reward)),0,max(0,(math.tanh(3*classes.space[i].state_reward))**2)))

    axes1.add_patch(square)


plt.savefig("maze.png")
plt.close()


sys.stdout=sys.__stdout__
for row in range(6):
    print(row)
    epsilon_policy=classes.policy(args.epsilon_greedy(0))
    init_state=random.randint(0,len(classes.space)-1)
    eps=classes.episode(init_state,epsilon_policy,-1)
    cnt=0
    flg=1
    for s,a in eps.track:
        if not flg:
            break
        fig1 = plt.figure(num=1, figsize=(max_column,max_row))
        plt.title(f"Action value:{classes.space[s].actions[a].action_value}")
        axes1 = fig1.add_subplot(1,1,1)
        if classes.space[s].state_reward>0:
            flg=0
        for i in range(len(classes.space)):
            y=i//max_column
            x=i%max_column
            #if not i%max_column:
            #    stdout.write("\n")
            #stdout.write(f"{state_value[i]:5.3f}\t")
            agent=0
            if i==s:
                for ac in classes.space[s].actions:
                    j=next(ac)
                    _y=j//max_column
                    _x=j%max_column
                    plt.text(1/max_column*(_x+0.5), 1/max_row*(_y+0.5),f"{ac.action_value:.3f}")
            if i==s or i==next(classes.space[s].actions[a]): #in list(map(lambda t:next(t),classes.space[s].actions)):
                agent=1
            square = plt.Rectangle(xy=(1/max_column*x, 1/max_row*y), width=1/max_column, height=1/max_row, alpha=0.8, angle=0.0,color=(max(0,-math.tanh(classes.space[i].state_reward)),agent,max(0,math.tanh(3*classes.space[i].state_reward))))

            axes1.add_patch(square)



        plt.savefig(f"track{row}\\"+str(cnt))
        cnt+=1
        plt.close()