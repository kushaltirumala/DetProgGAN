import numpy as np 
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init

from model_utils import *

def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(-1, keepdim=True)

def num_trainable_params(model):
    total = 0
    for p in model.parameters():
        count = 1
        for s in p.size():
            count *= s
        total += count
    return total

def cudafy_list(states):
    for i in range(len(states)):
        states[i] = states[i].cuda()
    return states

class Discriminator(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, gpu=True, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gpu = gpu
        self.num_layers = num_layers 
        self.name = "regular"
        hidden_dim2 = hidden_dim

        self.gru = nn.GRU(state_dim, hidden_dim, num_layers)
        self.dense1 = nn.Linear(hidden_dim + action_dim, hidden_dim2)
        self.dense2 = nn.Linear(hidden_dim2, 1)
        
    def forward(self, x, a, h=None):  # x: seq * batch * 10, a: seq * batch * 10
        p, hidden = self.gru(x, h)   # p: seq * batch * 10
        p = torch.cat([p, a], 2)   # p: seq * batch * 20
        prob = F.sigmoid(self.dense2(F.relu(self.dense1(p))))    # prob: seq * batch * 1
        return prob

    def init_hidden(self, batch):
        return Variable(torch.zeros(self.num_layers, batch, self.hidden_dim))

class Discriminator_minibatch(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, gpu=True, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gpu = gpu
        self.num_layers = num_layers 
        self.name = "regular"
        hidden_dim2 = hidden_dim

        self.gru = nn.GRU(state_dim, hidden_dim, num_layers)
        self.dense1 = nn.Linear(hidden_dim + action_dim, hidden_dim2)
        self.dense2 = nn.Linear(2 * hidden_dim2, 1)
        
        # minibatch_discriminator
        self.mean = True
        self.T = nn.Parameter(torch.Tensor(hidden_dim2, hidden_dim2, 96))
        init.normal(self.T, 0, 1)
        #self.dense = nn.Linear(hidden_dim + action_dim, 1)
            
    def forward(self, x, a, h=None):  # x: seq * batch * 22, a: seq * batch * 22
        p, hidden = self.gru(x, h)   # p: seq * batch * 22
        p = torch.cat([p, a], 2)   # p: seq * batch * 44
        fc1 = F.relu(self.dense1(p))
        fc2 = []
        for i in range(x.shape[0]):
            fc2.append(self.batch_discrim(fc1[i]))
        prob = F.sigmoid(self.dense2(torch.stack(fc2, 0)))    # prob: seq * batch * 1
        #prob = F.sigmoid(self.dense(p))
        return prob

    def batch_discrim(self, x):
        # x is NxA
        # T is AxBxC
        N, A = x.shape
        _, B, C = self.T.shape
        
        matrices = x.mm(self.T.view(A, -1))
        matrices = matrices.view(-1, B, C)

        M = matrices.unsqueeze(0)  # 1xNxBxC
        M_T = M.permute(1, 0, 2, 3)  # Nx1xBxC
        norm = torch.abs(M - M_T).sum(3)  # NxNxB
        expnorm = torch.exp(-norm)
        o_b = (expnorm.sum(0) - 1)   # NxB, subtract self distance
        if self.mean:
            o_b /= N - 1

        x = torch.cat([x, o_b], 1)
        return x

    def init_hidden(self, batch):
        return Variable(torch.zeros(self.num_layers, batch, self.hidden_dim))
        
class Discriminator_entire(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, gpu=True, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gpu = gpu
        self.num_layers = num_layers
        self.name = "entire"

        self.gru = nn.GRU(state_dim, hidden_dim, num_layers)
        self.dense1 = nn.Linear(hidden_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, 1)
            
    def forward(self, x, a, h=None):  # x: seq * batch * 10, a: seq * batch * 10
        x1 = torch.cat([x, a[-1:, :, :]], 0)
        p, hidden = self.gru(x1, h)   # p: seq * batch * h
        prob = F.sigmoid(self.dense2(F.relu(self.dense1(p))))    # prob: seq * batch * 1
        return prob[-1, :, :]

    def init_hidden(self, batch):
        return Variable(torch.zeros(self.num_layers, batch, self.hidden_dim))

class Discriminator_nosig(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, gpu=True, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gpu = gpu
        self.num_layers = num_layers 
        hidden_dim2 = hidden_dim
        self.name = "nosig"

        self.gru = nn.GRU(state_dim, hidden_dim, num_layers)
        self.dense1 = nn.Linear(hidden_dim + action_dim, hidden_dim2)
        self.dense2 = nn.Linear(hidden_dim2, 1)
        #self.dense = nn.Linear(hidden_dim + action_dim, 1)
            
    def forward(self, x, a, h=None):  # x: seq * batch * 22, a: seq * batch * 22
        p, hidden = self.gru(x, h)   # p: seq * batch * 22
        p = torch.cat([p, a], 2)   # p: seq * batch * 44
        output = self.dense2(F.relu(self.dense1(p)))    # output: seq * batch * 1
        #output = self.dense(p)
        return output

    def init_hidden(self, batch):
        return Variable(torch.zeros(self.num_layers, batch, self.hidden_dim))

class Value(nn.Module):
    def __init__(self, state_dim, hidden_dim, gpu=True, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.gpu = gpu
        self.num_layers = num_layers 

        self.gru = nn.GRU(state_dim, hidden_dim, num_layers)
        self.dense1 = nn.Linear(hidden_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, 1)
            
    def forward(self, x, h=None, test=False):  # seq * batch * 10
        value_ori, hidden = self.gru(x, h)
        value = self.dense2(F.relu(self.dense1(value_ori)))    # seq * batch * 1
        
        if not test:
            return value, hidden
        else:
            return value, hidden, value_ori

    def init_hidden(self, batch):
        return Variable(torch.zeros(self.num_layers, batch, self.hidden_dim))

class PROG_RNN_DET(nn.Module):

    def __init__(self, params):
        super(PROG_RNN_DET, self).__init__()

        self.params = params
        x_dim = params['x_dim']
        y_dim = params['y_dim']
        h_dim = params['h_dim']
        m_dim = params['m_dim']
        rnn_micro_dim = params['rnn_micro_dim']
        rnn_mid_dim = params['rnn_mid_dim']
        rnn_macro_dim = params['rnn_macro_dim']
        n_layers = params['n_layers']
        n_agents = params['n_agents']
        
        self.gru_macro = nn.GRU(y_dim, rnn_macro_dim, n_layers)
        self.dec_macro = nn.Sequential(
            nn.Linear(rnn_macro_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, y_dim))
        
        self.gru_mid = nn.GRU(y_dim, rnn_mid_dim, n_layers)
        self.dec_mid = nn.Sequential(
            nn.Linear(rnn_mid_dim + y_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, y_dim))
            
        self.gru_micro = nn.GRU(y_dim, rnn_micro_dim, n_layers)
        self.dec_micro = nn.Sequential(
            nn.Linear(rnn_micro_dim + y_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, y_dim))

    def forward(self, data, hp=None):
        # data: seq_length * batch * 10
        loss = 0
        h_macro = Variable(torch.zeros(self.params['n_layers'], data.size(1), self.params['rnn_macro_dim']))
        h_mid = Variable(torch.zeros(self.params['n_layers'], data.size(1), self.params['rnn_mid_dim']))
        h_micro = Variable(torch.zeros(self.params['n_layers'], data.size(1), self.params['rnn_micro_dim']))
        if self.params['cuda']:
            h_macro, h_mid, h_micro = h_macro.cuda(), h_mid.cuda(), h_micro.cuda()

        if hp['train'] == 'macro':
            for t in range(data.shape[0] - 1):
                state_t = data[t].clone()
                next_t = data[t+1].clone()
                
                _, h_macro = self.gru_macro(state_t.unsqueeze(0), h_macro)
                dec_t = self.dec_macro(h_macro[-1])
                
                loss += torch.sum((dec_t - next_t) ** 2)
        
        elif hp['train'] == 'mid':
            for t in range(0, data.shape[0] - 2, 2):
                state_t = data[t].clone()
                next_t = data[t+1].clone()

                remainder = int(t/16) + 1

                macro_t = data[4*remainder].clone()
                
                _, h_mid = self.gru_mid(state_t.unsqueeze(0), h_mid)
                dec_t = self.dec_mid(torch.cat([h_mid[-1], macro_t], 1))

                loss += torch.sum((dec_t - next_t) ** 2)
                
                _, h_mid = self.gru_mid(next_t.unsqueeze(0), h_mid)
    
        elif hp['train'] == 'micro':
            for t in range(0, data.shape[0] - 2, 2):
                state_t = data[t].clone()
                next_t = data[t+1].clone()
                mid_t = data[t+2].clone()
                
                _, h_micro = self.gru_micro(state_t.unsqueeze(0), h_micro)
                dec_t = self.dec_micro(torch.cat([h_micro[-1], mid_t], 1))

                loss += torch.sum((dec_t - next_t) ** 2)
                
                _, h_micro = self.gru_micro(next_t.unsqueeze(0), h_micro)    
    
        return loss
    
    def sample_macro(self, data, burn_in=0):
        h_macro = Variable(torch.zeros(self.params['n_layers'], data.size(1), self.params['rnn_macro_dim']))
        if self.params['cuda']:
            h_macro = h_macro.cuda()
        
        ret = []
        state_t = data[0]
        ret.append(state_t)  

        for t in range(data.shape[0] - 1):
            _, h_macro = self.gru_macro(state_t.unsqueeze(0), h_macro)
            dec_t = self.dec_macro(h_macro[-1])
            state_t = dec_t
            ret.append(state_t)
            
        return torch.stack(ret, 0)

    def sample(self, data, burn_in=0):
        # data: seq_length * batch * 10
        h_macro = Variable(torch.zeros(self.params['n_layers'], data.size(1), self.params['rnn_macro_dim']))
        h_mid = Variable(torch.zeros(self.params['n_layers'], data.size(1), self.params['rnn_mid_dim']))
        if self.params['cuda']:
            h_macro, h_mid = h_macro.cuda(), h_mid.cuda()
        
        ret = []
        state_t = data[0]
        ret.append(state_t)  

        for t in range(0, data.shape[0] - 2, 2):
            _, h_macro = self.gru_macro(state_t.unsqueeze(0), h_macro)
            dec_t = self.dec_macro(h_macro[-1])
            macro_t = dec_t
            
            _, h_mid = self.gru_mid(state_t.unsqueeze(0), h_mid)
            dec_t = self.dec_mid(torch.cat([h_mid[-1], macro_t], 1))
            state_t = dec_t
            ret.append(state_t)
            ret.append(macro_t)
            
            _, h_mid = self.gru_mid(state_t.unsqueeze(0), h_mid)
            state_t = macro_t
            
        return torch.stack(ret, 0)
    
    def sample_micro(self, data, burn_in=0):
        # data: seq_length * batch * 10
        h_macro = Variable(torch.zeros(self.params['n_layers'], data.size(1), self.params['rnn_macro_dim']))
        h_mid = Variable(torch.zeros(self.params['n_layers'], data.size(1), self.params['rnn_mid_dim']))
        h_micro = Variable(torch.zeros(self.params['n_layers'], data.size(1), self.params['rnn_micro_dim']))
        if self.params['cuda']:
            h_macro, h_mid, h_micro = h_macro.cuda(), h_mid.cuda(), h_micro.cuda()
        
        seq_len = data.shape[0]
        ret = []
        state_t = data[0]
        ret.append(state_t)
        
        for t in range(0, seq_len, 4):
            # generate macro
            _, h_macro = self.gru_macro(state_t.unsqueeze(0), h_macro)
            dec_t = self.dec_macro(h_macro[-1])
            macro_t = dec_t
            if t + 4 <= burn_in:
                macro_t = data[t + 4]
            
            # generate mid
            _, h_mid = self.gru_mid(state_t.unsqueeze(0), h_mid)
            dec_t = self.dec_mid(torch.cat([h_mid[-1], macro_t], 1))
            mid_t = dec_t 
            if t + 2 <= burn_in:
                mid_t = data[t + 2]
            _, h_mid = self.gru_mid(mid_t.unsqueeze(0), h_mid)
            
            # generate micro
            _, h_micro = self.gru_micro(state_t.unsqueeze(0), h_micro)
            dec_t = self.dec_micro(torch.cat([h_micro[-1], mid_t], 1))
            state_t = dec_t
            if t + 1 <= burn_in:
                state_t = data[t + 1]
            ret.append(state_t)
            _, h_micro = self.gru_micro(state_t.unsqueeze(0), h_micro)
            state_t = mid_t
            ret.append(state_t)
            _, h_micro = self.gru_micro(state_t.unsqueeze(0), h_micro)
            dec_t = self.dec_micro(torch.cat([h_micro[-1], macro_t], 1))
            state_t = dec_t
            if t + 3 <= burn_in:
                state_t = data[t + 3]
            ret.append(state_t)
            _, h_micro = self.gru_micro(state_t.unsqueeze(0), h_micro)
            state_t = macro_t
            ret.append(state_t)
            
        return torch.stack(ret, 0)[:seq_len]

    def stop_grad_helper(self):
        print("stop gru_macro:")
        for p in self.gru_macro.parameters():
            print(p.shape)
            p.requires_grad=False
        print("stop dec_macro:")
        for p in self.dec_macro.parameters():
            print(p.shape)
            p.requires_grad=False
        
        print("stop gru_mid:")
        for p in self.gru_mid.parameters():
            print(p.shape)
            p.requires_grad=False
        print("stop dec_mid:")
        for p in self.dec_mid.parameters():
            print(p.shape)
            p.requires_grad=False  
