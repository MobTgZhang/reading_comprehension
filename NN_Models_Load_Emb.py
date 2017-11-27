import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter
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

class Com_CNN_RNN(nn.Module):
	def __init__(self,vocab_size,embedding_dim,hidden_dim,temp_dim = 50,out_dim = 2,num_layers = 80,
				num_epochs = 2,algorithmA = "RNN",algorithmB = "RNN"):
		super(Com_CNN_RNN,self).__init__()
		self.embedding = nn.Embedding(vocab_size,embedding_dim)
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		self.num_epochs = num_epochs
		self.algorithmA = algorithmA.strip()
		self.algorithmB = algorithmB.strip()
		# The recurrent neural network
		self.rnnA = None
		if algorithmA == "RNN" or algorithmA == "rnn":
			self.rnnA = nn.RNN(embedding_dim,embedding_dim,num_layers)
		elif algorithmA == "GRU" or algorithmA == "gru":
			self.rnnA = nn.GRU(embedding_dim,embedding_dim,num_layers)
		elif algorithmA == "LSTM" or algorithmA == "lstm":
			self.rnnA = nn.LSTM(embedding_dim,embedding_dim,num_layers)
		else:
			raise Exception("Unknown TypeA :" + self.algorithmA)

		kernel_size = self.embedding_dim
		stride = 2
		dilation = 1
		padding = math.floor(dilation * (kernel_size - 1)/2)
		self.con_out_dim = (self.embedding_dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
		self.conv = nn.Conv1d(self.num_layers,self.num_layers,kernel_size,stride,padding,dilation)
		stride = 2
		dilation = 1
		padding = math.floor(dilation * (kernel_size - 1)/2)
		self.max_pool_out_dim = math.floor((self.con_out_dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
		self.maxpool = nn.MaxPool1d(kernel_size,stride,padding,dilation)

		self.rnnB = None
		if algorithmB == "RNN" or algorithmB == "rnn":
			self.rnnB = nn.RNN(self.max_pool_out_dim,hidden_dim,1)
		elif algorithmB == "GRU" or algorithmB == "gru":
			self.rnnB = nn.GRU(self.max_pool_out_dim,hidden_dim,1)
		elif algorithmB == "LSTM" or algorithmB == "lstm":
			self.rnnB = nn.LSTM(self.max_pool_out_dim,hidden_dim,1)
		else:
			raise Exception("Unknown TypeA :" + self.algorithmB)

		self.double_linear = Double_Linear(self.hidden_dim,self.hidden_dim,temp_dim)
		self.linear = nn.Linear(temp_dim,out_dim)

		self.hidden_rnnA = self.init_hidden(self.num_layers,algorithmA,self.embedding_dim)
		self.hidden_rnnB = self.init_hidden(1,algorithmB,self.hidden_dim)
	def forward(self,sentA,sentB):
		hA = self.embedding(sentA)
		hB = self.embedding(sentB)
		# recurent neural network
		for k in range(self.num_epochs):
			hA = self.run_rnnA(hA)
			hB = self.run_rnnA(hB)
		hA = self.run_conv(hA)
		hB = self.run_conv(hB)

		_,hA = self.rnnB(hA,self.hidden_rnnB)
		_,hB = self.rnnB(hB,self.hidden_rnnB)
		
		hA = hA.view(len(hA),self.hidden_dim)
		hB = hB.view(len(hB),self.hidden_dim)
		hx = hA *hB
		hv = torch.abs(hA - hB)
		hs = self.double_linear(hx,hv)
		ht = F.tanh(hs)
		ht = self.linear(ht)
		return F.softmax(ht)
	def run_conv(self,sentence):
		hn = sentence.view(1,self.num_layers,self.embedding_dim)
		hn = self.conv(hn)
		hn = self.maxpool(hn)
		hn = hn.view(self.num_layers,1,self.max_pool_out_dim)
		return hn
	def run_rnnA(self,sentence):
		_,hn = self.rnnA(sentence.view(len(sentence),1,self.embedding_dim),self.hidden_rnnA)
		if self.algorithmB == "RNN" or self.algorithmB == "rnn":
			pass
		elif self.algorithmB == "GRU" or self.algorithmB == "gru":
			pass
		elif self.algorithmB == "LSTM" or self.algorithmB == "lstm":
			hn = hn[0]
		else:
			raise Exception("Unknown TypeA :" + self.algorithmB)
		return hn
	def init_hidden(self,num_layers,algorithm,out_dim):
		if algorithm == "RNN" or algorithm == "rnn":
			return Variable(torch.randn(num_layers,1,out_dim))
		elif algorithm == "GRU" or algorithm == "gru":
			return Variable(torch.randn(num_layers,1,out_dim))
		elif algorithm == "LSTM" or algorithm == "lstm":
			return (Variable(torch.randn(num_layers,1,out_dim)),
				Variable(torch.randn(num_layers,1,out_dim)))
		else:
			raise Exception("Unknown TypeA :" + algorithm)
	def initialize_hidden_layer(self):
		self.hidden_rnnA = self.init_hidden(self.num_layers,self.algorithmA,self.embedding_dim)
		self.hidden_rnnB = self.init_hidden(1,self.algorithmB,self.hidden_dim)
class Sim_CNN_RNN(nn.Module):
	def __init__(self,vocab_size,embedding_dim,hidden_dim,out_dim,num_layers_n = 80,
				kernel_size = 3,stride = 1,padding = 1,dilation = 1,algorithm = "RNN"):
		super(Sim_CNN_RNN,self).__init__()
		self.num_layers = num_layers_n
		self.hidden_dim = hidden_dim
		self.embedding_dim = embedding_dim
		self.algorithm = algorithm
		self.embedding = nn.Embedding(vocab_size,embedding_dim)#num_layers = num_layers_n
		# RNN Layer A
		self.rnnA = nn.RNN(embedding_dim,hidden_dim, num_layers=num_layers_n)
		# Convolution Layer 
		padding = math.floor(dilation * (hidden_dim - 1)/2)
		self.con_out_dim = math.floor((hidden_dim  + 2 * padding - dilation * (hidden_dim - 1) - 1) / stride + 1)
		self.conv1d = nn.Conv1d(num_layers_n,num_layers_n,hidden_dim,stride,padding,dilation)
		# MaxPooling Layer
		stride = 1
		dilation = 1
		delta = 13
		padding = math.floor(dilation * (self.con_out_dim - 1)/2) - delta
		self.maxpool1d = nn.MaxPool1d(self.con_out_dim,stride,padding,dilation)
		# RNN Layer B
		self.max_pool_out_dim = math.floor((self.con_out_dim  + 2 * padding - dilation * (self.con_out_dim - 1) - 1) / stride + 1)
		self.rnnB = nn.RNN(self.max_pool_out_dim,self.max_pool_out_dim)
		# initialize the hidden layer
		self.hidden_rnnA = self.init_hidden_rnnA()
		self.hidden_rnnB = self.init_hidden_rnnB()
		# double Linear 
		self.temp_dim = 5
		self.double_linear = Double_Linear(self.max_pool_out_dim,self.max_pool_out_dim,self.temp_dim)
		self.linear = nn.Linear(self.temp_dim ,out_dim)
	def forward(self,sentA,sentB):
		outA = self.run_model(sentA)
		outB = self.run_model(sentB)
		return self.hidden_setup(outA,outB)
	def init_hidden_rnnA(self):
		return Variable(torch.randn(self.num_layers,1,self.hidden_dim))
	def init_hidden_rnnB(self):
		return Variable(torch.randn(1,1,self.max_pool_out_dim))
	def run_model(self,sentence):
		out_q = self.embedding(sentence).view(len(sentence),1,self.embedding_dim)
		_,hn = self.rnnA(out_q,self.hidden_rnnA)
		hn = hn.squeeze().view(1,self.num_layers,self.hidden_dim)
		hn = self.conv1d(hn)
		out_q = self.maxpool1d(hn)
		out_q = out_q.squeeze()
		out_q = out_q.view(len(out_q),1,self.max_pool_out_dim)
		_,hn = self.rnnB(out_q,self.hidden_rnnB)
		hn = hn.view(1,self.max_pool_out_dim)
		del out_q
		return hn
	def hidden_setup(self,sentA,sentB):
		hx = sentA * sentB
		hv = torch.abs(sentA - sentB)
		ht = self.double_linear(hx,hv)
		ht = F.tanh(ht)
		ht = self.linear(ht)
		return F.softmax(ht)
	def initialize_hidden_layer(self):
		self.hidden_rnnA = self.init_hidden_rnnA()
		self.hidden_rnnB = self.init_hidden_rnnB()

class Simple_RNN(nn.Module):
	def __init__(self,vocab_size,embedding_dim,hidden_dim,temp_dim,out_dim,dp = 0.2,
				algorithm = "RNN",
				bi = True):
		super(Simple_RNN,self).__init__()
		# Embedding Layer defination
		self.embedding_dim = embedding_dim
		self.embedding = nn.Embedding(vocab_size,embedding_dim)
		# hidden dimension defination
		self.hidden_dim = hidden_dim
		self.linear = Double_Linear(hidden_dim,hidden_dim,temp_dim)
		self.out_linear = nn.Linear(temp_dim,out_dim)
		# The defination of algorithm
		self.RNN_Model = None
		self.algorithm = algorithm.strip()
		if algorithm == "RNN" or self.algorithm == "rnn":
			self.RNN_Model = nn.RNN(embedding_dim,hidden_dim,
				num_layers = 1,
				bias = bi,
				dropout = dp)
		elif algorithm == "LSTM" or self.algorithm == "lstm":
			self.RNN_Model = nn.LSTM(embedding_dim,hidden_dim,
				num_layers = 1,
				bias = bi,
				dropout = dp)
		elif self.algorithm == "GRU" or self.algorithm == "gru":
			self.RNN_Model = nn.GRU(embedding_dim,hidden_dim,
				num_layers = 1,
				bias = bi,
				dropout = dp)
		else:
			raise Exception("Unknow model"+algorithm)
		self.hidden = self.init_hidden()
	def forward(self,sentA,sentB):
		hnA = self.run_model(sentA)
		hnB = self.run_model(sentB)
		hv = self.linear(hnA,hnB)
		ht = F.tanh(hv)
		ht = self.out_linear(ht)
		return F.softmax(ht)
	def run_model(self,sentence):
		out_q = self.embedding(sentence)
		_,hn = self.RNN_Model(out_q.view(len(out_q),1,self.embedding_dim),self.hidden)
		if self.algorithm == "LSTM" or self.algorithm == "lstm":
			hn = hn[0].view(1,self.hidden_dim)
		else:
			hn = hn.view(1,self.hidden_dim)
		del out_q
		return hn
	def init_hidden(self):
		if self.algorithm == "RNN" or self.algorithm == "rnn":
			return Variable(torch.randn(1,1,self.hidden_dim))
		elif self.algorithm == "LSTM" or self.algorithm == "lstm":
			return (Variable(torch.randn(1,1,self.hidden_dim)),
				Variable(torch.randn(1,1,self.hidden_dim)))
		elif self.algorithm == "GRU" or self.algorithm == "gru":
			return Variable(torch.randn(1,1,self.hidden_dim))
		else:
			raise Exception("Unknow model"+self.algorithm)
	def initialize_hidden_layer(self):
		self.hidden = self.init_hidden()
class Linear_RNN(nn.Module):
	def __init__(self,vocab_size,embedding_dim,hidden_dim,out_dim,
				dp = 0.2,num_layers = 10,
				algorithm = "RNN",
				bi = True):
		super(Linear_RNN,self).__init__()
		self.embedding_dim = embedding_dim
		self.algorithm = algorithm
		self.num_layers = num_layers

		self.embedding = nn.Embedding(vocab_size,embedding_dim)


		#self.linear = nn.Linear(hidden_dim,embedding_dim)
		self.out_linear = nn.Linear(hidden_dim,out_dim)
		self.double_linear = Double_Linear(embedding_dim,embedding_dim,hidden_dim)


		# The defination of algorithm
		self.RNN_Model = None
		self.algorithm =algorithm.strip()
		if algorithm == "RNN" or self.algorithm == "rnn":
			self.RNN_Model = nn.RNN(embedding_dim,embedding_dim,
				num_layers = 1,
				bias = bi,
				dropout = dp)
		elif algorithm == "LSTM" or self.algorithm == "lstm":
			self.RNN_Model = nn.LSTM(embedding_dim,embedding_dim,
				num_layers = 1,
				bias = bi,
				dropout = dp)
		elif self.algorithm == "GRU" or self.algorithm == "gru":
			self.RNN_Model = nn.GRU(embedding_dim,embedding_dim,
				num_layers = 1,
				bias = bi,
				dropout = dp)
		else:
			raise Exception("Unknow model"+algorithm)
		self.hidden = self.init_hidden()
	def forward(self,sentA,sentB):
		hnA = self.embedding(sentA)
		hnB = self.embedding(sentB)
		for k in range(self.num_layers):
			hnA = self.run_model(hnA)
			hnB = self.run_model(hnB)
		hv = self.double_linear(hnA,hnB)
		ht = F.tanh(hv)
		ht = self.out_linear(ht)
		return F.softmax(ht)
	def run_model(self,out_q):
		out_q = F.softplus(out_q)
		_,hn = self.RNN_Model(out_q.view(len(out_q),1,self.embedding_dim),self.hidden)
		if self.algorithm == "LSTM" or self.algorithm == "lstm":
			hn = hn[0].view(1,self.embedding_dim)
		else:
			hn = hn.view(1,self.embedding_dim)
		return hn
	def init_hidden(self):
		if self.algorithm == "RNN" or self.algorithm == "rnn":
			return Variable(torch.randn(1,1,self.embedding_dim))
		elif self.algorithm == "LSTM" or self.algorithm == "lstm":
			return (Variable(torch.randn(1,1,self.embedding_dim)),
				Variable(torch.randn(1,1,self.embedding_dim)))
		elif self.algorithm == "GRU" or self.algorithm == "gru":
			return Variable(torch.randn(1,1,self.embedding_dim))
		else:
			raise Exception("Unknow model"+self.algorithm)
	def initialize_hidden_layer(self):
		self.hidden = self.init_hidden()

class Complex_RNN(nn.Module):
	def __init__(self,vocab_size,embedding_dim,hidden_dim,out_dim,
				dp = 0.2,num_layers = 10,
				algorithm = "RNN",
				bi = True):
		super(Complex_RNN,self).__init__()
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.algorithm = algorithm.strip()
		self.num_layers = num_layers
		self.embedding = nn.Embedding(vocab_size,embedding_dim)

		self.double_linear = Double_Linear(embedding_dim,embedding_dim,hidden_dim)
		self.out_linear = nn.Linear(hidden_dim,out_dim)
		# The defination of algorithm
		self.RNN_Model = None
		if algorithm == "RNN" or self.algorithm == "rnn":
			self.RNN_Model = nn.RNN(embedding_dim,embedding_dim,
				num_layers = 1,
				bias = bi,
				dropout = dp)
		elif algorithm == "LSTM" or self.algorithm == "lstm":
			self.RNN_Model = nn.LSTM(embedding_dim,embedding_dim,
				num_layers = 1,
				bias = bi,
				dropout = dp)
		elif self.algorithm == "GRU" or self.algorithm == "gru":
			self.RNN_Model = nn.GRU(embedding_dim,embedding_dim,
				num_layers = 1,
				bias = bi,
				dropout = dp)
		else:
			raise Exception("Unknow model"+algorithm)
		self.hidden = self.init_hidden()
	def forward(self,sentA,sentB):
		hnA = self.run_model(sentA)
		hnB = self.run_model(sentB)
		
		hx = hnA*hnB
		hv = torch.abs(hnA - hnB)
		hs = self.double_linear(hx,hv)
		ht = F.tanh(hs)
		ht = self.out_linear(ht)
		return F.softmax(ht)
	def run_model(self,sentence):
		hn = self.embedding(sentence)
		for k in range(self.num_layers):
			hn = self.iter_sect(hn)
		return hn
	def iter_sect(self,out_q):
		out_q = F.softplus(out_q)
		_,hn = self.RNN_Model(out_q.view(len(out_q),1,self.embedding_dim),self.hidden)
		if self.algorithm == "LSTM" or self.algorithm == "lstm":
			hn = hn[0].view(1,self.embedding_dim)
		else:
			hn = hn.view(1,self.embedding_dim)
		return hn
	def init_hidden(self):
		if self.algorithm == "RNN" or self.algorithm == "rnn":
			return Variable(torch.randn(1,1,self.embedding_dim))
		elif self.algorithm == "LSTM" or self.algorithm == "lstm":
			return (Variable(torch.randn(1,1,self.embedding_dim)),
				Variable(torch.randn(1,1,self.embedding_dim)))
		elif self.algorithm == "GRU" or self.algorithm == "gru":
			return Variable(torch.randn(1,1,self.embedding_dim))
		else:
			raise Exception("Unknow model"+self.algorithm)
	def initialize_hidden_layer(self):
		self.hidden = self.init_hidden()
