import multiprocessing.managers
import multiprocessing.queues
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn as nn
import multiprocessing
import random
import args
import classes
import numpy as np
import copy
import time
Net_copy=multiprocessing.Queue(5)

def Target_Net_Processing(in_Pipe,export_net,queue_copy,stop_flag):
    time.sleep(1)
    #print(2)
    raw_net=classes.QN(3,1)
    net=classes.QN(3,1).to(args.device)
    flag=1
    while flag:
        while in_Pipe.empty():
            #print(f"stop_flag in TN{stop_flag.value}")
            if stop_flag.value:
                flag=0
                break
            time.sleep(0.01)
        if not flag:
            break
        try:
            mini_batch = in_Pipe.get() #mini_batch=[((row,column,action),action_value)]
        except EOFError:
            return 0
        timer_start=time.time()
        if not net.train(mini_batch):
            break
        period=time.time()-timer_start
        print(f"train time {period}")
        #print("training happended")
        print(f"is copy queue full: {queue_copy.full()}")
        if not queue_copy.full():
            queue_copy.put(copy.deepcopy(net.to("cpu")),block=0)
            net.to(args.device)
    #export_net.put(copy.deepcopy(net))
    net.to("cpu")
    export_net.put(net)
    print("NT_Processing over")
def Active_Net_Processing(queue_sa,queue_out,queue_copy,stop_flag):
    net=classes.QN(3,1).to(args.device)
    flag=True
    batch=[]
    round_cnt=0
    while flag:
        
        got=0
        #print(4)
        while not got:
            try:
                sa_pair=queue_sa.get(block=0)
                got=1
            except queue.Empty:
                if stop_flag.value:
                    print("AN over")
                    return
                continue
        for sa in sa_pair:
            #print(f"sa: {sa}")
            out=np.float64(net.forward(torch.tensor(sa).to(device=args.device).detach()).detach())
            #print(f"net outcome: {out}")
            batch.append(out)
        
        queue_out.put(batch)
        round_cnt+=1
        if round_cnt>(args.update_round)*args.batch_length:
            round_cnt=0
            try:
                net=queue_copy.get(block=0)
                net.to(args.device)
            except EOFError:
                queue_out.close()
                break
            #queue_copy=multiprocessing.Queue()
            except queue.Empty:
                print(__name__,"Empty raised")
                if stop_flag.value:
                    print("AN over")
                    return
                continue
    print("AN over")
    return
def Sample_Processing(space,max_row,max_column,queue_out,handle,queue_copy,stop_flag,treading_num=args.sample_threading_num):
    classes.space=space
    args.max_column=max_column
    args.max_row=max_row
    state_num=len(space)
    global cache
    cache=[]
    global cache_init
    cache_init=1
    def sub_processing():
        global cache_init
        global cache
        #print(3)
        #manager=multiprocessing.Manager()
        #Net_in_queue=manager.Queue(10)
        #Net_out_queue=manager.Queue(10)
        Net_in_queue=multiprocessing.Queue(10)
        Net_out_queue=multiprocessing.Queue(10)
        net_processing=multiprocessing.Process(target=Active_Net_Processing,args=(Net_in_queue,Net_out_queue,queue_copy,stop_flag))
        net_processing.start()
        #print(5)
        epsilon_policy=classes.policy(args.epsilon_greedy(1))
        ##print(5)
        ##print(f"SP:{stop_flag.value}")
        ##print(f"AN:{stop_flag.value}")
        ##print(f"TN:{args.stop_flag_TN.value}")
        while not stop_flag.value:
            out=[]
            #print(7)
            #print(f"cache length in thread {len(cache)}")
            init_state=random.randint(0,state_num-1)
            #print(f"init_state {init_state}")
            #print(f"state_num {len(classes.space)}")
            eps=classes.episode(init_state,epsilon_policy,args.episode_length)
            for ind in range(len(eps.track)-2,0,-1):
                s,a=eps.track[ind]
                x=s%args.max_column
                y=s//args.max_column
                next_s=classes.space[eps.track[ind+1][0]]
                sa_pairs=list(map(lambda a:(np.float32(x),np.float32(y),np.float32(a)),range(len(next_s.actions))))
                while Net_in_queue.full():
                    if stop_flag.value:
                        print("sample sub processing over")
                        return
                    time.sleep(0.05)
                Net_in_queue.put(sa_pairs)
                while Net_out_queue.empty():
                    if stop_flag.value:
                        print("sample sub processing over")
                        return
                    time.sleep(0.05)
                out.append(((np.float32(x),np.float32(y),np.float32(a)),np.float32(space[s].actions[a].action_reward+args.gamma*max(Net_out_queue.get()))))
            if cache_init:
                cache+=random.sample(out,len(out))
                if len(cache)>args.AN_cache_min_length:
                    cache_init=0
                continue

            #print(f"cache:{len(cache)}")
            #print(f"out:{len(out)}")

            position=random.sample(range(len(cache)),len(out))
            for i in range(len(out)):
                cache[position[i]]=out[i]
        print("sample sub processing over")
    pool=ThreadPoolExecutor(treading_num)
    threadings=[]
    for i in range(treading_num):
        #print("threading add")
        threadings.append(threading.Thread(target=sub_processing))
        threadings[i].start()
    #    threadings.append(pool.submit(sub_processing,i))
    while 1:
        try:
            if not handle.get() or stop_flag.value:
                handle.close()
                queue_out.close()
                break
        except EOFError:
            handle.close()
            queue_out.close()
            break
        while len(cache)<args.batch_length:
            #print(f"cache length{len(cache)}")
            time.sleep(0.5)
        queue_out.put(random.sample(cache,args.batch_length))
    print("sample processing over")


    






