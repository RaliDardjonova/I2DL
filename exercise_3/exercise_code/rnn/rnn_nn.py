import torch
import torch.nn as nn
import math

class RNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=20, activation="tanh"):
        super().__init__()
        """
        Inputs:
        - input_size: Number of features in input vector
        - hidden_size: Dimension of hidden vector
        - activation: Nonlinearity in cell; 'tanh' or 'relu'
        """
        #######################################################################
        # TODO: Build a simple one layer RNN with an activation with the      #
        # attributes defined above and a forward function below. Use the      #
        # nn.Linear() function as your linear layers.                         #
        # Initialse h as 0 if these values are not given.                     #
        #######################################################################
        self.hidden_size = hidden_size
        self.activation = activation
        self.V = nn.Parameter(torch.Tensor(input_size, hidden_size)) # 4 X 1
        self.W = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) # 1 X 1
        self.b = nn.Parameter(torch.Tensor(hidden_size))
        self.b2 = nn.Parameter(torch.Tensor(hidden_size))

        #self.i2h = nn.Linear(input_size + hidden_size, hidden_size)

        self.init_weights()
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x, h=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Optional hidden vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence
                 (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq = []
        #######################################################################
        #                                YOUR CODE                            #
        #######################################################################
        seq_len, batch_size, input_size = x.shape
        if h is None:
            h = torch.zeros(batch_size, self.hidden_size)

        for t in range(seq_len):
            #combined = torch.cat((x[t],h), 1)
            #h = self.i2h(combined)
            h = torch.tanh(h @ self.W + x[t] @ self.V + self.b + self.b2)
            h_seq.append(h.unsqueeze(1))

        h_seq = torch.cat(h_seq)
        h = h.unsqueeze(1)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        return h_seq, h

def rnn_tanh_cell(input, hidden, w_ih, w_hh, b_ih, b_hh):
    igates = torch.mm(input, w_ih.t()) + b_ih
    hgates = torch.mm(hidden, w_hh.t()) + b_hh
    return torch.tanh(igates + hgates)

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=20, p=1):
        super().__init__()
        #######################################################################
        # TODO: Build a one layer LSTM with an activation with the attributes #
        # defined above and a forward function below. Use the                 #
        # nn.Linear() function as your linear layers.                         #
        # Initialse h and c as 0 if these values are not given.               #
        #######################################################################
        self.hidden_size = hidden_size

        #self.W_ii = nn.Parameter(torch.Tensor(input_size, hidden_size))
        #self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        #self.b_i = nn.Parameter(torch.Tensor(hidden_size))
        # forget gate
        #self.W_if = nn.Parameter(torch.Tensor(input_size, hidden_size))
        #self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        #self.b_f = nn.Parameter(torch.Tensor(hidden_size))
        # ???
        #self.W_ig = nn.Parameter(torch.Tensor(input_size, hidden_size))
        #self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        #self.b_g = nn.Parameter(torch.Tensor(hidden_size))
        # output gate
        #self.W_io = nn.Parameter(torch.Tensor(input_size, hidden_size))
        #self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        #self.b_o = nn.Parameter(torch.Tensor(hidden_size))
        self.V = nn.Parameter(torch.Tensor(4*hidden_size, input_size)) # 4 X 1
        self.W = nn.Parameter(torch.Tensor(4*hidden_size, hidden_size)) # 1 X 1
        self.b = nn.Parameter(torch.Tensor(4*hidden_size))
        self.b2 = nn.Parameter(torch.Tensor(4*hidden_size))

        self.init_weights()


        len = self.b.shape[0]
        print(len)
        for i in range(len):
            if i >= (0.25)*len and i < (0.5)*len:
                #self.b[i] = p
                pass

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def lstm_cell(self, input, h_x, c_x, w_ih, w_hh, b_ih, b_hh):
    # type: (Tensor, Tuple[Tensor, Tensor], Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
        hx = h_x
        cx = c_x
        gates = torch.mm(input, w_ih.t()) + torch.mm(hx, w_hh.t()) + b_ih + b_hh
        #print(b_ih,b_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, cy



    def forward(self, x, h=None, c=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Hidden vector (nr_layers, batch_size, hidden_size)
        - c: Cell state vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence
                 (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        - c: Final cell state vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq = []
        #######################################################################
        #                                YOUR CODE                            #
        #######################################################################
        seq_len, batch_size, input_size = x.shape
        if h is None:
            h_t = torch.zeros(batch_size,self.hidden_size).to(x.device)

        if c is None:
            c_t = torch.zeros(batch_size,self.hidden_size).to(x.device)

        for t in range(seq_len): # iterate over the time steps
            x_t = x[t]

            h_t, c_t = self.lstm_cell(x[t], h_t, c_t, self.V, self.W,self.b, self.b2)
            h_seq.append(h_t.unsqueeze(1))

        h_seq = torch.cat(h_seq)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        #hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        h = h_t.unsqueeze(1)
        c = c_t.unsqueeze(1)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        return h_seq, (h, c)


class RNN_Classifier(torch.nn.Module):
    def __init__(self, classes=10, input_size=28, hidden_size=128,
                 activation="relu"):
        super(RNN_Classifier, self).__init__()
        #######################################################################
        #  TODO: Build a RNN classifier                                       #
        #######################################################################
        self.classes = classes
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(input_size, hidden_size, nonlinearity=activation)
        self.dense = nn.Linear(hidden_size, self.classes)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def init_hidden(self,):
        # (num_layers, batch_size, n_neurons)
        return (torch.zeros(1, self.batch_size, self.hidden_size))

    def forward(self, X):
        # transforms X to dimensions: n_steps X batch_size X n_inputs
        #X = X.permute(1, 0, 2)

        self.batch_size = X.size(1)
        self.hidden = self.init_hidden()

        lstm_out, self.hidden = self.rnn(X, self.hidden)
        out = self.dense(self.hidden)

        return out.view(-1, self.classes)

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)


class LSTM_Classifier(torch.nn.Module):
    def __init__(self, classes=10, input_size=28, hidden_size=128):
        super(LSTM_Classifier, self).__init__()
        #######################################################################
        #  TODO: Build a LSTM classifier                                      #
        #######################################################################
        self.classes = classes
        self.hidden_size = hidden_size

        self.rnn = nn.LSTM(input_size, hidden_size)
        self.dense = nn.Linear(hidden_size, self.classes)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, X):

        self.batch_size = X.size(1)
        hidden = (torch.zeros(1, self.batch_size, self.hidden_size))

        lstm_out, hidden = self.rnn(X, hidden)
        out = self.dense(hidden)

        return out.view(-1, self.classes)

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
