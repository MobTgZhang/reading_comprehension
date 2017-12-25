import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import copy
import math

class Double_Linear(nn.Module):
    def __init__(self,in_dimA,in_dimB,out_feature,bias = True):
        super(Double_Linear,self).__init__()
        self.in_dimA = in_dimA
        self.in_dimB = in_dimB
        self.out_feature = out_feature
        self.weightA = Parameter(torch.randn(in_dimA,out_feature))
        self.weightB = Parameter(torch.randn(in_dimB,out_feature))
        if bias:
            self.bias = Parameter(torch.randn(out_feature))
        else:
            self.reset_parameter('bias',None)
        self.reset_parameters()
    def reset_parameters(self):
        stdvA = 1./math.sqrt(self.weightA.size(1))
        stdvB = 1./math.sqrt(self.weightB.size(1))
        self.weightA.data.uniform_(-stdvA,stdvA)
        self.weightB.data.uniform_(-stdvB,stdvB)
        if self.bias is not None:
            self.bias.data.uniform_(-stdvB,stdvB)
    def forward(self,inputA,inputB):
        return F.linear(inputA,self.weightA.t(),None)+F.linear(inputB,self.weightB.t(),self.bias)
    def __repr__(self):
        return self.__class__.__name__ + ' ( ('\
            + str(self.in_dimA) + ' , '\
            + str(self.in_dimB) + ' ) ->'\
            + str(self.out_feature) + ' ) '
# Simple RNN model with no attention
class SimpleRNN(nn.Module):
    def __init__(self,params,emb_matrix = None):
        super(SimpleRNN,self).__init__()
        # parameters for model
        self.vocab_size = params['vocab_size']
        self.embedding_dim =  params['embedding_dim']
        self.hidden_dim = params['hidden_dim']
        self.out_dim = params['out_dim']
        self.bidirectional = params['bidirectional']
        self.algorithm = params['algorithm'].lower()
        # embedding layers
        self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim)
        if emb_matrix is not None:
            self.embedding.weight.data = emb_matrix
        # RNN layer
        if self.algorithm == 'lstm':
            self.rnn = nn.LSTM(self.embedding_dim,self.hidden_dim,num_layers=1,
                               dropout=0.5,bidirectional=self.bidirectional,batch_first=True)
        elif self.algorithm == "rnn":
            self.rnn = nn.RNN(self.embedding_dim,self.hidden_dim,num_layers=1,
                              dropout=0.5,bidirectional=self.bidirectional,batch_first=True)
        elif self.algorithm == "gru":
            self.rnn = nn.GRU(self.embedding_dim,self.hidden_dim,num_layers=1,
                              dropout=0.5,bidirectional=self.bidirectional,batch_first=True)
        else:
            raise (NameError,"Unknown Type:" + self.algorithm)
        # Bidirectional weights layer
        num_dir = 2 if self.bidirectional else 1
        self.bi_weight = nn.Linear(num_dir,1)
        # Bi-Linear
        self.bilinear = Double_Linear(self.hidden_dim,self.hidden_dim,self.hidden_dim)
        # out Layer
        self.out = nn.Linear(self.hidden_dim,self.out_dim)
        # sigmoid layer
        self.sigmoid = nn.Sigmoid()
        # initialize weights
        self.initial_weights()
    def forward(self,sentA,sentB,hidden):
        embA = self.embedding(sentA)
        embB = self.embedding(sentB)
        embA = embA.view(1,len(sentA),self.embedding_dim)
        embB = embB.view(1,len(sentB),self.embedding_dim)

        _,hidA = self.rnn(embA)
        _,hidB = self.rnn(embB)
        if self.algorithm == 'lstm':
            hidA = hidA[0]
            hidB = hidB[0]
        # bi-Linear
        if self.bidirectional:
            hidA = self.bi_weight(hidA.squeeze().t()).t()
            hidB = self.bi_weight(hidB.squeeze().t()).t()
        else:
            hidA = hidA.squeeze().view(1,hidA.size(2))
            hidB = hidB.squeeze().view(1,hidB.size(2))
        biout = self.bilinear(hidA,hidB)
        out_l = self.out(F.tanh(biout))
        return self.sigmoid(out_l)
    def initial_hidden(self):
        num_dir = 2 if self.bidirectional else 1
        if self.algorithm == 'lstm':
            return (Variable(torch.zeros(1,num_dir,self.hidden_dim)),
                    Variable(torch.zeros(1,num_dir,self.hidden_dim)))
        elif self.algorithm == 'gru' or self.algorithm == 'rnn':
            return Variable(torch.zeros(1,num_dir,self.hidden_dim))
        else:
            raise (NameError, "Unknown Type:" + self.algorithm)
    def initial_weights(self):
        initrange = 0.1
        initrange_w = 0.5
        self.bilinear.weightA.data.uniform_(-initrange,initrange)
        self.bilinear.weightB.data.uniform_(-initrange,initrange)
        self.out.weight.data.uniform_(-initrange,initrange)
        self.bi_weight.weight.data.uniform_(-initrange_w,initrange_w)
# Simple RNN with attention
class AttSimpleRNN(nn.Module):
    def __init__(self,params,emb_matrix = None):
        super(AttSimpleRNN,self).__init__()
        # parameters for model
        self.vocab_size = params['vocab_size']
        self.embedding_dim = params['embedding_dim']
        self.hidden_dim = params['hidden_dim']
        self.temp_dim = params['temp_dim']
        self.out_dim = params['out_dim']
        self.algorithm = params['algorithm'].lower()
        self.bidirectional = params['bidirectional']

        # layer for the RNN
        self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim)
        if emb_matrix is not None:
            self.embedding.weight.data = emb_matrix
        # RNN layer
        if self.algorithm == 'lstm':
            self.rnn = nn.LSTM(self.embedding_dim,self.hidden_dim,num_layers=1,
                               dropout=0.5,bidirectional=self.bidirectional,batch_first=True)
        elif self.algorithm == "rnn":
            self.rnn = nn.RNN(self.embedding_dim,self.hidden_dim,num_layers=1,
                              dropout=0.5,bidirectional=self.bidirectional,batch_first=True)
        elif self.algorithm == "gru":
            self.rnn = nn.GRU(self.embedding_dim,self.hidden_dim,num_layers=1,
                              dropout=0.5,bidirectional=self.bidirectional,batch_first=True)
        else:
            raise (NameError,"Unknown Type:" + self.algorithm)
        # Bidirectional weights layer
        if self.bidirectional:
            self.bi_weight = nn.Linear(2, 1)
        # bi-linear
        self.bi_linear = Double_Linear(self.hidden_dim,self.hidden_dim,self.temp_dim)
        # tanh layer
        self.tanh_layer = nn.Tanh()
        # temp hidden layer
        self.linear = nn.Linear(self.temp_dim,self.out_dim)
        # sigmoid layer
        self.sigmoid = nn.Sigmoid()
        # initialize the hidden layer weights
        self.initial_weights()
    def forward(self,sentA,sentB,hidden):
        embA = self.embedding(sentA)
        embB = self.embedding(sentB)
        embA = embA.view(1,len(sentA),self.embedding_dim)
        embB = embB.view(1,len(sentB),self.embedding_dim)

        # hidden layer for rnn
        _,hidA = self.rnn(embA,hidden)
        _,hidB = self.rnn(embB,hidden)
        if self.algorithm == 'lstm':
            hidA = hidA[0]
            hidB = hidB[0]
        # bi-Linear
        if self.bidirectional:
            hidA = self.bi_weight(hidA.squeeze().t()).t()
            hidB = self.bi_weight(hidB.squeeze().t()).t()
        else:
            hidA = hidA.squeeze().view(1, hidA.size(2))
            hidB = hidB.squeeze().view(1, hidB.size(2))

        ht = hidA * hidB
        hv = torch.abs(hidA - hidB)
        ht = self.bi_linear(ht,hv)

        hv = self.tanh_layer(ht)
        ht = self.linear(hv)
        return self.sigmoid(ht)

    def initial_hidden(self):
        num_dir = 2 if self.bidirectional else 1
        if self.algorithm == 'lstm':
            return (Variable(torch.zeros(1, num_dir, self.hidden_dim)),
                    Variable(torch.zeros(1, num_dir, self.hidden_dim)))
        elif self.algorithm == 'gru' or self.algorithm == 'rnn':
            return Variable(torch.zeros(1, num_dir, self.hidden_dim))
        else:
            raise (NameError, "Unknown Type:" + self.algorithm)
    def initial_weights(self):
        initrange = 0.1
        # Bidirectional weights layer
        if self.bidirectional:
            initrange_w = 0.5
            self.bi_weight.weight.data.uniform_(-initrange_w,initrange_w)
        # bi-linear
        self.bi_linear.weightA.data.uniform_(-initrange,initrange)
        self.bi_linear.weightB.data.uniform_(-initrange,initrange)
        # temp hidden layer
        self.linear.weight.data.uniform_(-initrange,initrange)
# Linear RNN with no attention
class LinearRNN(nn.Module):
    def __init__(self,params,emb_matrix = None):
        super(LinearRNN,self).__init__()
        # parameters for model
        self.embedding_dim = params['embedding_dim']
        self.vocab_size = params['vocab_size']
        self.hidden_dim = params['hidden_dim']
        self.num_layers = params['num_layers']
        self.temp_dim = params['temp_dim']
        self.out_dim = params['out_dim']
        self.algorithm = params['algorithm'].lower()

        # Embedding Layer
        self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim)
        if emb_matrix is not None:
            self.embedding.weight.data = emb_matrix
        # RNN layers for the model
        # RNN layer
        if self.algorithm == 'lstm':
            self.rnn = nn.LSTM(self.embedding_dim, self.embedding_dim, num_layers=1,
                               dropout=0.5, bidirectional=False, batch_first=True)
        elif self.algorithm == "rnn":
            self.rnn = nn.RNN(self.embedding_dim, self.embedding_dim, num_layers=1,
                              dropout=0.5, bidirectional=False, batch_first=True)
        elif self.algorithm == "gru":
            self.rnn = nn.GRU(self.embedding_dim, self.embedding_dim, num_layers=1,
                              dropout=0.5, bidirectional=False, batch_first=True)
        else:
            raise (NameError, "Unknown Type:" + self.algorithm)
        # linear Temporary layer
        self.Temp_hidden = nn.Linear(self.embedding_dim,self.hidden_dim)
        # SoftPlus Layer
        self.soft_plus_layer = nn.Softplus(beta=1.0)
        # bi_layer
        self.bi_linear = Double_Linear(self.hidden_dim,self.hidden_dim,self.temp_dim)
        # tanh_layer
        self.tanh_layer = nn.Tanh()
        # out layer
        self.out_linear = nn.Linear(self.temp_dim,self.out_dim)
        # sigmoid layer
        self.sigmoid = nn.Sigmoid()
    def forward(self,sentA,sentB,hidden):
        embA = self.embedding(sentA)
        embB = self.embedding(sentB)
        outA = embA.view(1, len(sentA), self.embedding_dim)
        outB = embB.view(1, len(sentB), self.embedding_dim)

        # stacked layer
        hidA = copy.deepcopy(hidden)
        hidB = copy.deepcopy(hidden)
        
        for k in range(self.num_layers):
            outA,hidA = self.rnn(outA,hidA)
            outB,hidB = self.rnn(outB,hidB)
        # hidden layer processing
        if self.algorithm == 'lstm':
            hidA = hidA[0]
            hidB = hidB[0]
        hidA = hidA.squeeze().view(1, hidA.size(2))
        hidB = hidB.squeeze().view(1, hidB.size(2))
        
        htA = self.soft_plus_layer(self.Temp_hidden(hidA))
        htB = self.soft_plus_layer(self.Temp_hidden(hidB))

        ht = self.bi_linear(htA,htB)
        ht = self.tanh_layer(ht)
        ht = self.out_linear(ht)
        return self.sigmoid(ht)
    def initial_hidden(self):
        if self.algorithm == 'lstm':
            return (Variable(torch.zeros( 1,1, self.embedding_dim)),
                    Variable(torch.zeros( 1,1, self.embedding_dim)))
        elif self.algorithm == 'gru' or self.algorithm == 'rnn':
            return Variable(torch.zeros( 1, 1,self.embedding_dim))
        else:
            raise (NameError, "Unknown Type:" + self.algorithm)
    def initial_weights(self):
        initrange = 0.1
        # Bidirectional weights layer
        if self.bidirectional:
            initrange_w = 0.5
            self.bi_weight.weight.data.uniform_(-initrange_w,initrange_w)
        # bi-linear
        self.Temp_hidden.weight.data.uniform_(-initrange,initrange)
        self.bi_linear.weight.data.uniform_(-initrange,initrange)
        # temp hidden layer
        self.out_linear.weight.data.uniform_(-initrange,initrange)
# Linear Model with attention
class AttLinearRNN(nn.Module):
    def __init__(self,params,emb_matrix = None):
        super(AttLinearRNN,self).__init__()
        # parameters for model
        self.embedding_dim = params['embedding_dim']
        self.vocab_size = params['vocab_size']
        self.hidden_dim = params['hidden_dim']
        self.num_layers = params['num_layers']
        self.temp_dim = params['temp_dim']
        self.out_dim = params['out_dim']
        self.algorithm = params['algorithm'].lower()

        # Embedding Layer
        self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim)
        if emb_matrix is not None:
            self.embedding.weight.data = emb_matrix
        # RNN layers for the model
        # RNN layer
        if self.algorithm == 'lstm':
            self.rnn = nn.LSTM(self.embedding_dim, self.embedding_dim, num_layers=1,
                               dropout=0.5, bidirectional=False, batch_first=True)
        elif self.algorithm == "rnn":
            self.rnn = nn.RNN(self.embedding_dim, self.embedding_dim, num_layers=1,
                              dropout=0.5, bidirectional=False, batch_first=True)
        elif self.algorithm == "gru":
            self.rnn = nn.GRU(self.embedding_dim, self.embedding_dim, num_layers=1,
                              dropout=0.5, bidirectional=False, batch_first=True)
        else:
            raise (NameError, "Unknown Type:" + self.algorithm)
        # linear Temporary layer
        self.Temp_hidden = nn.Linear(self.embedding_dim, self.hidden_dim)
        # SoftPlus Layer
        self.soft_plus_layer = nn.Softplus(beta=1.0)
        self.bi_linear = Double_Linear(self.hidden_dim,self.hidden_dim,self.temp_dim)
        self.Tanh_layer = nn.Tanh()
        self.out_linear = nn.Linear(self.temp_dim,self.out_dim)
        self.sigmoid = nn.Sigmoid()
    def forward(self,sentA,sentB,hidden):
        embA = self.embedding(sentA)
        embB = self.embedding(sentB)
        outA = embA.view(1, len(sentA), self.embedding_dim)
        outB = embB.view(1, len(sentB), self.embedding_dim)

        # stacked layer
        hidA = copy.deepcopy(hidden)
        hidB = copy.deepcopy(hidden)
        for k in range(self.num_layers):
            outA, hidA = self.rnn(outA, hidA)
            outB, hidB = self.rnn(outB, hidB)
        # hidden layer processing
        if self.algorithm == 'lstm':
            hidA = hidA[0]
            hidB = hidB[0]
        hidA = hidA.squeeze().view(1, hidA.size(2))
        hidB = hidB.squeeze().view(1, hidB.size(2))
        htA = self.soft_plus_layer(self.Temp_hidden(hidA))
        htB = self.soft_plus_layer(self.Temp_hidden(hidB))

        ht = htA * htB
        hv = torch.abs(htA - htB)
        hm = self.bi_linear(ht,hv)
        hv = self.out_linear(self.Tanh_layer(hm))
        return self.sigmoid(hv)
    def initial_hidden(self):
        if self.algorithm == 'lstm':
            return (Variable(torch.zeros(1, 1, self.embedding_dim)),
                    Variable(torch.zeros(1, 1, self.embedding_dim)))
        elif self.algorithm == 'gru' or self.algorithm == 'rnn':
            return Variable(torch.zeros(1, 1, self.embedding_dim))
        else:
            raise (NameError, "Unknown Type:" + self.algorithm)
    def initial_weights(self):
        initrange = 0.1
        # Bidirectional weights layer
        if self.bidirectional:
            initrange_w = 0.5
            self.bi_weight.weight.data.uniform_(-initrange_w,initrange_w)
        # bi-linear
        self.bi_linear.weight.data.uniform_(-initrange,initrange)
        self.Temp_hidden.weight.data.uniform_(-initrange,initrange)
        # temp hidden layer
        self.out_linear.weight.data.uniform_(-initrange,initrange)

# Complex RNN Model for sentence
class MultiRNN(nn.Module):
    def __init__(self,params,emb_matrix = None):
        super(MultiRNN,self).__init__()
        # parameters for the model
        self.vocab_size = params['vocab_size']
        self.embedding_dim = params['embedding_dim']
        self.hidden_dim = params['hidden_dim']
        self.out_dim = params['out_dim']
        self.num_layers = params['num_layers']
        self.temp_dim = params['temp_dim']
        self.algorithm = params['algorithm'].lower()
        # Embedding Layer
        self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim)
        if emb_matrix is not None:
            self.embedding.weight.data = emb_matrix
        # RNN Layer
        if self.algorithm == "lstm":
            self.rnn = nn.LSTM(self.embedding_dim,self.embedding_dim,num_layers=1,
                               bidirectional=False,dropout=0.5,batch_first=True)
        elif self.algorithm == "gru":
            self.rnn = nn.GRU(self.embedding_dim,self.embedding_dim,num_layers=1,
                              bidirectional=False,dropout=0.5,batch_first=True)
        elif self.algorithm == "rnn":
            self.rnn = nn.RNN(self.embedding_dim,self.embedding_dim,num_layers=1,
                              bidirectional=False,dropout=0.5,batch_first=True)
        else:
            raise (NameError,"Unknow Type:" + self.algorithm)
        # Bilinear layer
        self.bi_Linear = Double_Linear(self.embedding_dim,self.embedding_dim,self.hidden_dim)
        # Tanh Layer
        self.tanh_layer = nn.Tanh()
        # linear layer
        self.out_linear = nn.Linear(self.hidden_dim,self.out_dim)
        # sigmoid layer
        self.sigmoid = nn.Sigmoid()
        self.initial_weights()
    def forward(self,sentA,sentB,hidden):
        hnA = self.run_model(sentA,hidden)
        hnB = self.run_model(sentB,hidden)

        hx = hnA * hnB
        hv = torch.abs(hnA - hnB)
        hs = self.bi_Linear(hx,hv)
        ht = self.tanh_layer(hs)
        ht = self.out_linear(ht)
        return self.sigmoid(ht)
    def run_model(self,sentence,hidden):
        hn = self.embedding(sentence)
        for k in range(self.num_layers):
            hn = self.iter_sect(hn,hidden)
        return hn
    def iter_sect(self,emb,hidden):
        out_q = F.softplus(emb)
        out_q = out_q.view(1,len(emb),self.embedding_dim)
        _,hn = self.rnn(out_q,hidden)
        if self.algorithm == "lstm":
            hn = hn[0].view(1,self.embedding_dim)
        else:
            hn = hn.view(1,self.embedding_dim)
        return hn
    def initial_hidden(self):
        if self.algorithm == 'lstm':
            return (Variable(torch.zeros(1, 1, self.embedding_dim)),
                    Variable(torch.zeros(1, 1, self.embedding_dim)))
        elif self.algorithm == 'gru' or self.algorithm == 'rnn':
            return Variable(torch.zeros(1, 1, self.embedding_dim))
        else:
            raise (NameError, "Unknown Type:" + self.algorithm)
    def initial_weights(self):
        initrange = 0.1
        self.bi_Linear.weightA.data.uniform_(-initrange,initrange)
        self.bi_Linear.weightB.data.uniform_(-initrange,initrange)
        self.out_linear.weight.data.uniform_(-initrange,initrange)
# Simple mixed CNN and RNN layer
class Sim_CNN_RNN(nn.Module):
    def __init__(self,params,emb_matrix = None):
        super(Sim_CNN_RNN,self).__init__()
        self.embedding_dim = params['embedding_dim']
        self.vocab_size = params['vocab_size']
        self.hidden_dim = params['hidden_dim']
        self.temp_dim = params['temp_dim']
        self.out_dim = params['out_dim']
        self.algorithm = params['algorithm']
        self.num_layers = params['num_layers']
        self.dilation = params['dilation']
        self.stride = params['stride']

        # Embedding Layer
        self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim)
        if emb_matrix is not None:
            self.embedding.weight.data = emb_matrix
        # RNN Layer A
        self.rnn_first = nn.GRU(self.embedding_dim,self.hidden_dim,self.num_layers,batch_first=True)
        # Convolution Layer
        padding = math.floor(self.dilation*(self.hidden_dim - 1)/2)
        self.con_out_dim = math.floor((self.hidden_dim + 2*padding - self.dilation*(self.hidden_dim-1)-1)/self.stride + 1)
        self.conv1d = nn.Conv1d (self.num_layers,self.num_layers,self.hidden_dim,self.stride,padding,self.dilation)
        # MaxPooling Layer
        stride = 1
        dilation = 1
        delta = 13
        padding = math.floor(dilation*(self.con_out_dim -1)/2 ) - delta
        self.maxpooling = nn.MaxPool1d(self.con_out_dim,stride,padding,dilation)
        self.max_pool_out_dim = math.floor((self.con_out_dim + 2*padding - dilation*(self.con_out_dim - 1)-1)/stride +1)
        # RNN Layer B
        self.rnn_second = nn.LSTM(self.max_pool_out_dim,self.max_pool_out_dim,num_layers=1,batch_first=True)
        self.bi_linear = Double_Linear(self.max_pool_out_dim,self.max_pool_out_dim,self.temp_dim)
        self.tanh_layer = nn.Tanh()
        self.linear = nn.Linear(self.temp_dim,self.out_dim)
        self.sigmoid = nn.Sigmoid()
        # Hidden Layer initialize
        self.hiddenGRU = None
        self.hiddenLSTM = None
        self.initial_weights()
    def forward(self,sentA,sentB,hidden):
        outA = self.run_model(sentA)
        outB = self.run_model(sentB)
        return self.hidden_setup(outA,outB)
    def run_model(self,sentence):
        out_q = self.embedding(sentence).view(1,len(sentence),self.embedding_dim)
        _,hn = self.rnn_first(out_q,self.hiddenGRU)
        hn = hn.squeeze().view(1,self.num_layers,self.hidden_dim)
        hn = self.conv1d(hn)
        out_q = self.maxpooling(hn)
        out_q = out_q.squeeze()
        out_q = out_q.view(1,len(out_q),self.max_pool_out_dim)
        _,hn = self.rnn_second(out_q,self.hiddenLSTM)
        hn  = hn[0].view(1,self.max_pool_out_dim)
        del out_q
        return hn
    def hidden_setup(self,sentA,sentB):
        hx = sentA * sentB
        hv = torch.abs(sentA - sentB)
        ht = self.bi_linear(hx,hv)
        ht = self.tanh_layer(ht)
        ht = self.linear(ht)
        return self.sigmoid(ht)
    def initial_hidden(self):
        self.hiddenGRU = Variable(torch.zeros(self.num_layers,1,self.hidden_dim))
        self.hiddenLSTM = (Variable(torch.zeros(1,1,self.max_pool_out_dim)),
                           Variable(torch.zeros(1,1,self.max_pool_out_dim)))
    def initial_weights(self):
        initrange = 0.1
        self.bi_linear.weightA.data.uniform_(-initrange,initrange)
        self.bi_linear.weightB.data.uniform_(-initrange,initrange)
        self.linear.weight.data.uniform_(-initrange,initrange)
# Multi-Stacked CNN and RNN layers
class Com_CNN_RNN(nn.Module):
    def __init__(self,params,emb_matrix = None):
        super(Com_CNN_RNN,self).__init__()
        # parameters
        self.embedding_dim = params['embedding_dim']
        self.vocab_size = params['vocab_size']
        self.hidden_dim = params['hidden_dim']
        self.temp_dim = params['temp_dim']
        self.num_layers = params['num_layers']
        self.num_epoches = params['num_epoches']
        self.algorithm_first = params['algorithmA'].lower()
        self.algorithm_second = params['algorithmB'].lower()
        self.out_dim = params['out_dim']

        # Embedding Layer
        self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim)
        # RNN Layer
        if self.algorithm_first == 'rnn':
            self.rnn_first = nn.RNN(self.embedding_dim,self.embedding_dim,self.num_layers,batch_first=True)
        elif self.algorithm_first == 'gru':
            self.rnn_first = nn.GRU(self.embedding_dim,self.embedding_dim,self.num_layers,batch_first=True)
        elif self.algorithm_first == 'lstm':
            self.rnn_first = nn.LSTM(self.embedding_dim,self.embedding_dim,self.num_layers,batch_first=True)
        else:
            raise (NameError,"Unknown Type:" + self.algorithm_first)
        kernel_size = self.embedding_dim
        stride = 2
        dilation = 1
        padding = math.floor(dilation * (kernel_size - 1)/2)
        self.con_out_dim = (self.embedding_dim + 2 * padding - dilation * (kernel_size - 1)-1)/stride + 1
        self.conv = nn.Conv1d(self.num_layers,self.num_layers,kernel_size,stride,padding,dilation)
        stride = 2
        dilation = 1
        padding = math.floor(dilation * (kernel_size - 1)/2)
        self.max_pool_out_dim = math.floor((self.con_out_dim + 2*padding - dilation * (kernel_size - 1) -1)/stride + 1)
        self.maxpool = nn.MaxPool1d(kernel_size,stride,padding,dilation)

        if self.algorithm_second == 'rnn':
            self.rnn_second = nn.RNN(self.max_pool_out_dim,self.hidden_dim,num_layers=1,batch_first=True)
        elif self.algorithm_second == 'lstm':
            self.rnn_second = nn.LSTM(self.max_pool_out_dim,self.hidden_dim,num_layers=1,batch_first=True)
        elif self.algorithm_second == 'gru':
            self.rnn_second = nn.GRU(self.max_pool_out_dim,self.hidden_dim,num_layers=1,batch_first=True)
        else:
            raise (NameError,"Unknown Type:" + self.algorithm_second)
        # bi_linear Layer
        self.bi_linear = Double_Linear(self.hidden_dim,self.hidden_dim,self.temp_dim)
        # Tanh Layer
        self.tanh_layer = nn.Tanh()
        # linear layer
        self.linear = nn.Linear(self.temp_dim,self.out_dim)
        # sigmoid layer
        self.sigmoid = nn.Sigmoid()
        # initial hidden layer
        self.hidden_first = None
        self.hidden_second = None
        self.initial_weights()
    def forward(self,sentA,sentB,hidden = None):
        hA = self.embedding(sentA)
        hB = self.embedding(sentB)
        for _ in range(self.num_epoches):
            hA = self.run_first_RNN(hA)
            hB = self.run_first_RNN(hB)
        hA = self.run_conv(hA)
        hB = self.run_conv(hB)

        _,hA = self.rnn_second(hA,self.hidden_second)
        _,hB = self.rnn_second(hB,self.hidden_second)

        if self.algorithm_second == 'lstm':
            hA = hA[0]
            hB = hB[0]
        hA = hA.view(1,self.hidden_dim)
        hB = hB.view(1,self.hidden_dim)
        hx = hA * hB
        hv = torch.abs(hA - hB)
        hs = self.bi_linear(hx,hv)
        ht = self.tanh_layer(hs)
        hs = self.linear(ht)
        return self.sigmoid(hs)
    def run_first_RNN(self,EmbSent):
        EmbSent = EmbSent.view(1,len(EmbSent),self.embedding_dim)
        _,hn = self.rnn_first(EmbSent,self.hidden_first)
        if self.algorithm_first == "lstm":
            hn = hn[0]
        return hn
    def run_conv(self,sentence):
        hn = sentence.view(1,self.num_layers,self.embedding_dim)
        hn = self.conv(hn)
        hn = self.maxpool(hn)
        hn = hn.view(1,self.num_layers,self.max_pool_out_dim)
        return hn
    def initial_hidden(self):
        # initial hidden layer
        if self.algorithm_first == 'lstm':
            self.hidden_first = (Variable(torch.zeros(self.num_layers,1,self.embedding_dim)),
                                 Variable(torch.zeros(self.num_layers,1,self.embedding_dim)))
        elif self.algorithm_first == 'gru' or self.algorithm_first == 'rnn':
            self.hidden_first = Variable(torch.zeros(self.num_layers,1,self.embedding_dim))
        else:
            raise (NameError,"Unknown Type:" + self.algorithm_first)
        if self.algorithm_second == 'lstm':
            self.hidden_second = (Variable(torch.zeros(1,1,self.hidden_dim)),
                                  Variable(torch.zeros(1,1,self.hidden_dim)))
        elif self.algorithm_second == 'gru' or self.algorithm_second == 'rnn':
            self.hidden_second = Variable(torch.zeros(1,1,self.hidden_dim))
        else:
            raise (NameError, "Unknown Type:" + self.algorithm_second)
    def initial_weights(self):
        initrange = 0.1
        self.bi_linear.weightA.data.uniform_(-initrange,initrange)
        self.bi_linear.weightB.data.uniform_(-initrange,initrange)
        self.linear.weight.data.uniform_(-initrange,initrange)

if __name__ == "__main__":
    params = {
        'vocab_size':400,
        'embedding_dim':300,
        'hidden_dim':200,
        'temp_dim':150,
        'out_dim':1,
        'bidirectional':True,
        'algorithm':'gru',
        'num_layers':10,
        'dilation':1,
        'stride':1
    }
    path = 'data'
    from data_pre import Corpus
    corpus = Corpus(path)
    Model = Sim_CNN_RNN(params)
    input_valA = Variable(torch.LongTensor([1,45,78,6,4,74,4]))
    input_valB = Variable(torch.LongTensor([1,4,78,5,85,6,25,74,4]))
    hidden = Model.initial_hidden()
    result = Model(input_valA,input_valB,hidden)
    print(result)