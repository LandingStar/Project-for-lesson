import torch
import torch.nn as nn
import multiprocessing
import random
import args
import numpy as np
class QN(nn.Module):
    def __init__(self, state_dim, action_dim,inp,out):
        super(QN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, action_dim)
        self.function =nn.Sigmoid()
        self.inp,self.out=multiprocessing.Pipe(True)
        self.criterion = torch.nn.BCELoss(reduction='mean')  # 返回损失的平均值
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)


    def forward(self, x):
        x = self.function(self.fc1(x))
        x = self.function(self.fc2(x))
        x = self.function(self.fc3(x))
        return self.function(self.fc4(x))
    def train(self):
        try:
            mini_batch = self.inp.recv()
        except EOFError:
            return 0
        batch_length=len(mini_batch)
        sa, action_value= zip(*mini_batch)
        #next_states = np.array(next_states)
        for epoch in range(batch_length):  
           pred = self(sa[epoch])  
           loss = self.criterion(pred, action_value[epoch])  
           self.optimizer.zero_grad()
           loss.backward()  
           self.optimizer.step()
        return 1
def Target_Net_Processing(in_Pipe,Pipe_AN):
    net=QN(3,1,in_Pipe,Pipe_AN)
    while net.train():
        Pipe_AN.send(net.copy())
    return net
def Active_Net_Procssing(in_Pipe,Pipe_TN):
    net=QN(3,1,in_Pipe,Pipe_TN)
    batch=[]
    while args.batch_length






