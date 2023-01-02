'''
Author: Zhengxiang (Jack) Wang 
GitHub: https://github.com/jaaack-wang
Website: https://jaaack-wang.eu.org
About: A general RNN module written in PyTorch.
Allows Simple RNN/GRU/LSTM, bidirectional RNN,
multilayered RNN, etc.
'''
import torch
import torch.nn as nn


class RNN(nn.Module):
    '''A general RNN module written in PyTorch. Allows 
    Simple RNN/GRU/LSTM, bidirectional RNN, multilayered RNN.
    '''
    def __init__(self, 
                 in_vocab_size, out_vocab_size, 
                 hidden_size, embd_dim, num_layers=1,
                 rnn_type="SRNN", device=None, bias=True,
                 dropout_rate=0.0, bidirectional=False, 
                 reduction_method=torch.sum):
        '''Arguments for initializing the RNN class:
        
        - in_vocab_size (int): input vocab size. 
        - out_vocab_size (int): output vocab size.
        - hidden_size (int): hidden state size.
        - embd_dim (int): embedding size. 
        - num_layers (int): number of layers.
        - rnn_type (str): RNN type, SRNN, GRU, or LSTM.
        - device (torch.device): cpu or cuda
        - bias (bool): whether to use the bias terms in hidden state.
        - dropout_rate (float): 0~1, applied to embedding layer.
        - bidirectional (bool): whether to use bidirectional RNN.
        - reduction_method (method): method to use to reduce the two 
          hidden states (backward and forward) produced by bidirectional
          RNN. Defaults to torch.sum. torch.mean can also be used.
        '''
        super(RNN, self).__init__()
        
        self.embedding = nn.Embedding(in_vocab_size, embd_dim)
        
        self.num_layers = num_layers
        self.rnn_type = rnn_type.upper()
        self.bidirectional = bidirectional
        if self.rnn_type == "GRU": rnn_ = nn.GRU
        elif self.rnn_type == "LSTM": rnn_ = nn.LSTM
        elif self.rnn_type == "SRNN": rnn_ = nn.RNN
        else: raise ValueError("Only supports SRNN, GRU, LSTM," \
                               " but {self.rnn_type} was given.")
        self.rnn = rnn_(embd_dim, 
                        hidden_size, 
                        num_layers,
                        bias=bias,
                        bidirectional=bidirectional)
        self.hidden_size = hidden_size
        self.last_cell, self.last_hidden = None, None
        
        if not device:
            cuda = torch.cuda.is_available()
            self.device = torch.device('cuda' if cuda else 'cpu')
        else:
            self.device = device

        self.reduce = reduction_method
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(hidden_size, out_vocab_size, bias=bias)
        
    def forward(self, X):      
        # X: (max input seq len, batch size)
        # embd: (max input seq len, batch size, embd dim)
        embd = self.dropout(self.embedding(X))   
        seq_len, batch_size = X.shape
        # outputs: (max input seq len, batch size, 
        #                     hidden size * num directions)
        # hidden: (num directions * num layers, batch size, hidden size)
        # cell: (num directions * num layers, batch size, hidden size)
        if self.rnn_type == "LSTM":
            outputs, (hidden, cell) = self.rnn(embd)
        else:
            outputs, hidden = self.rnn(embd)
            cell = None # placeholder
         
        if self.bidirectional:
            hidden = hidden.view(2, self.num_layers, batch_size, -1)
            hidden = self.reduce(hidden, dim=0)
            
            if self.rnn_type == "LSTM":
                cell = cell.view(2, self.num_layers, batch_size, -1)
                cell = self.reduce(cell, dim=0)
            
            outputs = outputs.view(seq_len, batch_size, 2, -1)
            outputs = self.reduce(outputs, dim=2)
        
        # outputs: (max input seq len, batch size, out vocab size)
        # hidden: (num layers, batch size, hidden size)
        # cell: (num layers, batch size, hidden size) or None
        self.last_cell, self.last_hidden = cell, hidden
        return self.fc_out(outputs)

    
    def _init_hidden_state(self):
        '''Initializes a hidden state with a zero vector.
        This is for single input only (i.e., batch size is 1).
        '''
        return torch.zeros((self.num_layers, 1, self.hidden_size))
    
    
    def next_output(self, x):
        '''Computes the next state output.
        
        Args:
            - x (tensor): a torch.tensor of int type with size (1,1).
        '''
        
        # embd: (1, 1, embd dim)
        embd = self.dropout(self.embedding(x)).unsqueeze(0)
        
        cell, hidden = self.last_cell, self.last_hidden
        if hidden == None or hidden.shape[1] != 1:
            hidden = self._init_hidden_state()
        
        if cell == None or cell.shape[1] != 1:
            if self.rnn_type == "LSTM":
                cell = self._init_hidden_state()
        
        # output: (1, 1, hidden size * num directions)
        # hidden: (num directions * num layers, 1, hidden size)
        # cell: (num directions * num layers, 1, hidden size)
        if self.rnn_type == "LSTM":
            output, (hidden, cell) = self.rnn(embd, (hidden, cell))
        else:
            output, hidden = self.rnn(embd, hidden)
        
        if self.bidirectional:
            # hidden: (num layers, 1, hidden size)
            hidden = hidden.view(2, self.num_layers, batch_size, -1)
            hidden = self.reduce(hidden, dim=0)
            
            if self.rnn_type == "LSTM":
                # cell: (num layers, 1, hidden size)
                cell = cell.view(2, self.num_layers, batch_size, -1)
                cell = self.reduce(cell, dim=0)
            
            # output: (1, 1, hidden size)
            output = output.view(2, self.num_layers, batch_size, -1)
            output = self.reduce(output, dim=0)
        
        # output: (1, 1, out vocab size)
        self.last_cell, self.last_hidden = cell, hidden
        return self.fc_out(output)
