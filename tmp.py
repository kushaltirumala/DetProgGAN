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

    def forward(self, data, hp=None):
        # data: seq_length * batch * 10
        loss = 0
        h_macro = Variable(torch.zeros(self.params['n_layers'], data.size(1), self.params['rnn_macro_dim']))
        h_mid = Variable(torch.zeros(self.params['n_layers'], data.size(1), self.params['rnn_mid_dim']))
        if self.params['cuda']:
            h_macro, h_mid = h_macro.cuda(), h_mid.cuda()

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
                macro_t = data[t+2].clone()
                
                _, h_mid = self.gru_mid(state_t.unsqueeze(0), h_mid)
                dec_t = self.dec_mid(torch.cat([h_mid[-1], macro_t], 1))

                loss += torch.sum((dec_t - next_t) ** 2)
                
                _, h_mid = self.gru_mid(next_t.unsqueeze(0), h_mid)
    
        return loss
    
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