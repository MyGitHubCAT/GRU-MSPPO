import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple,deque
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
#device = ("cuda" 
#          if torch.cuda.is_available() 
#          else "mps" 
#          if torch.backends.mps.is_available() 
#          else "cpu")

class GRU_price(torch.nn.Module):
    def __init__(self):
        super(GRU_price,self).__init__()
        self.gru = torch.nn.GRU(input_size = 1,hidden_size = 128,batch_first = True)
        self.line1 = torch.nn.Linear(128,64)
        self.line2 = torch.nn.Linear(64,1)
    def forward(self,x):
        x = x.float()
        x.to(device)
        gru1,_ = self.gru(x)
        pre_gru1 = gru1.squeeze(-1)[:,-1]
        line1 = self.line1(pre_gru1)
        res = self.line2(line1)
        return res
    

class PPORR(torch.nn.Module):
    def __init__(self,inpdim,er_dim=5*5+1,init_pvratio_id = 1):
        super(PPORR,self).__init__()
        self.inputlayer = torch.nn.Linear(inpdim,64)
        self.linear1 = torch.nn.Linear(64,128)
        self.linear2 = torch.nn.Linear(128,64)

        self.out1 = torch.nn.Linear(64,er_dim)
        self.out3 = torch.nn.Linear(64,1)
        self.softmax = torch.nn.Softmax(dim=-1)
        

    def forward(self,inp):
        x = inp.to(device)
        inp1 = self.inputlayer(x)
        tanh1 = torch.nn.Tanh()(inp1)
        line1 = self.linear1(tanh1)
      
        line2 = self.linear2(line1)#
        tanh2 = torch.nn.Tanh()(line2)
        #
        out_1 = self.out1(tanh2)

        out_3 = self.out3(tanh2)
        res1 = self.softmax(out_1)

        res3 = out_3
        return res1,res3
    


class PPORP(torch.nn.Module):
    def __init__(self,inpdim,init_pvratio_id = 1):
        super(PPORP,self).__init__()
        self.inputlayer = torch.nn.Linear(inpdim,64)
        self.linear1 = torch.nn.Linear(64,128)
        self.linear2 = torch.nn.Linear(128,64)

        self.out2 = torch.nn.Linear(64,2)
        self.out3 = torch.nn.Linear(64,1)
        self.softmax = torch.nn.Softmax(dim=-1)      


    def forward(self,inp):
        x = inp.to(device)
        inp1 = self.inputlayer(x)
        tanh1 = torch.nn.Tanh()(inp1)
        line1 = self.linear1(tanh1)
      
        line2 = self.linear2(line1)#
        tanh2 = torch.nn.Tanh()(line2)

        out_2 = self.out2(tanh2)
        out_3 = self.out3(tanh2)

        res2 = self.softmax(out_2)
        res3 = out_3
        return res2,res3




BSSTransition = namedtuple('BSSTransition',('state_pool','value_pool','policy_ratio_pool','clipres_pool','log_prob_pool','reward_pool','done_pool'))#


class BSSMemory(object):#
    def __init__(self,capacity):
        self.memory = deque([],maxlen = capacity)
    def push(self,*args):
        self.memory.append(BSSTransition(*args))
    def sample(self):
        batch = list(self.memory)
        return zip(*batch)
    def __len__(self):
        return len(self.memory)
    def clear(self):
        self.memory.clear()


class BSS_DRL(object):#
    def __init__(self,Opt,inpdim=6,erdim=5*5+1,ifstep = True ):#
        self.options =  Opt 
                
        self.bss_RR = PPORR(inpdim,er_dim=erdim).to(device)#EVRP_DQN(5,TOTAL_POS_NUM)
        self.bss_RP = PPORP(inpdim).to(device)#EVRP_DQN(5,TOTAL_POS_NUM)   
        
    #
    def select_action(self,bssstate):#          
        prob1,state_value1 = self.bss_RR(bssstate)    
        prob2,state_value2 = self.bss_RP(bssstate)  
        energy_dist = torch.distributions.categorical.Categorical(prob1)
        pvratio_dist = torch.distributions.categorical.Categorical(prob2)
        
        energy_act = energy_dist.sample()
        pv_act =  pvratio_dist.sample()          
        return energy_act,pv_act,prob1,prob2 
   

