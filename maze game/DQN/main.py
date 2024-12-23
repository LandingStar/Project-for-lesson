import Dqn
import classes
import args
import surroundings_function
import multiprocessing
import time
import numpy as np
import torch
import copy
if __name__=='__main__':
    time.sleep(1)
    classes.space=surroundings_function.import_maze("C:\\Users\\16329\\source\\repos\\LandingStar\\CST-Project\\maze game\\epsilon-greedy\\mazz.txt")
    print(len(classes.space))
    print(args.max_row,args.max_column)
    #Queue_Sample_to_TN=multiprocessing.Queue(3)
    Pipe_Sample_to_TN_out,Pipe_Sample_to_TN_in=multiprocessing.Pipe(0)
    Queue_Sample_to_TN=multiprocessing.Queue(3)
    #handle_recv,handle_send=multiprocessing.Pipe(0)
    Queue_copy_net=multiprocessing.Queue(3)
    handle=multiprocessing.Queue(maxsize=1)
    net_export_out,net_export_in=multiprocessing.Pipe(0)
    net_export=multiprocessing.Queue(1)
    stop_flag=multiprocessing.Value("i",0)

    Target_Net_Processing=multiprocessing.Process(target=Dqn.Target_Net_Processing,args=(Queue_Sample_to_TN,net_export,Queue_copy_net,stop_flag))
    Sample_Processing=multiprocessing.Process(target=Dqn.Sample_Processing,args=(classes.space,args.max_row,args.max_column,Queue_Sample_to_TN,handle,Queue_copy_net,stop_flag))
    Sample_Processing.start()
    Target_Net_Processing.start()

    net_export_in.close()

    for i in range(args.run_round):
        handle.put(1)
        print(i,flush=1)
    stop_flag.value=1
    handle.put(0)
    outcome_net=net_export.get()
    outcome_net.to(args.device)
    handle.close()
    Target_Net_Processing.terminate()
    Target_Net_Processing.join()
    Target_Net_Processing.close()
    print("net got")
    state_num=len(classes.space)
    for s in range(state_num):
        for a in range(len(classes.space[s].actions)):
            x=np.float32(s%args.max_column)
            y=np.float32(s//args.max_column)
            classes.space[s].actions[a].action_value=np.float64(outcome_net.forward(torch.tensor((x,y,np.float32(a))).to(args.device)).detach().item())
            print(f"{s}_{a}/{state_num},action_value:{classes.space[s].actions[a].action_value}")
    surroundings_function.plot_maze(classes.space)
