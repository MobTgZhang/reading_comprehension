import torch
import time
from torch.autograd import Variable
import pandas as pd
# Make a dictionary
class Dictionary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
    def add_word(self,word):
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        return self.word2idx[word]
    def load_txt(self,path):
        train_data = pd.read_csv(path,sep="\t")
        Length = len(train_data)
        for index in range(Length):
            sentA = train_data.iloc[index]['sentence_A'].split()
            for word in sentA:
                self.add_word(word)
            sentB = train_data.iloc[index]['sentence_B'].split()
            for word in sentB:
                self.add_word(word)
    def __len__(self):
        return len(self.idx2word)
# sentence to index sequence
def seq_to_index(sentence,word_to_index):
    sentence = sentence.split()
    ids = [word_to_index[word] for word in sentence]
    tensor = torch.LongTensor(ids)
    return Variable(tensor)
def train_iter(model,train_data,dictionary_data,optimizer,loss_function):
    temp_list = []
    Length = len(train_data)
    for index in range(Length):
        sentA = seq_to_index(train_data.iloc[index]['sentence_A'],dictionary_data.word2idx)
        sentB = seq_to_index(train_data.iloc[index]['sentence_B'],dictionary_data.word2idx)
        score = (float(train_data.iloc[index]['relatedness_score']) - 1)/4
        target_score = Variable(torch.FloatTensor([score]).view(1, 1))
        optimizer.zero_grad()
        hidden = model.initial_hidden()
        predict_score = model(sentA,sentB,hidden)
        loss = loss_function(predict_score,target_score)
        loss.backward()
        optimizer.step()
        temp_list.append(loss.data[0])
        # print(train_data.iloc[index]['pair_ID'],index,loss.data[0])
        temp_list.append(loss.data[0])
    return temp_list
# training process
def train_model(model,train_data,dictionary_data,optimizer,loss_function,params):
    epoches = params['epoches']
    log_interval = params['interval']
    Length = len(train_data)
    all_losses = []
    start_time = time.time()
    for epoch in range(1):
        temp_data = train_data.iloc[200*1:200*(1+1)]
        batch_size = len(train_data)//200
        for k in range(batch_size):
            temp_data = train_data.iloc[200*k:200*(k+1)]
            temp_list = train_iter(model,temp_data,dictionary_data,optimizer,loss_function)
            print("%0.2f"%sum(temp_list)/len(temp_list))
    torch.save(model,params['model_name'])
    return all_losses
if __name__ == '__main__':
    dictionary_data = Dictionary()
    dictionary_data.load_txt('train.txt')
    dictionary_data.load_txt('test.txt')
    params = {
        'vocab_size': len(dictionary_data),
        'embedding_dim': 300,
        'hidden_dim': 200,
        'temp_dim':150,
        'out_dim': 1,
        'bidirectional': True,
        'algorithm': 'gru',
        'num_layers':10,
        'learning_rate':1e-3,
        'interval':10,
        'epoches':5000,
        'num_epoches':10,
        'stride':1,
        'algorithmA':'lstm',
        'algorithmB':'lstm',
        'dilation':1
    }
    import torch.nn as nn
    import torch.optim as optim
    from model import SimpleRNN,LinearRNN,AttLinearRNN,MultiRNN,Com_CNN_RNN,AttSimpleRNN,Sim_CNN_RNN
    path = 'data'
    Model = SimpleRNN(params)
    params['model_name'] = Model.__class__.__name__
    optimizer = optim.Adam(Model.parameters(),params['learning_rate'])
    loss_function= nn.BCELoss()
    train_data = pd.read_csv('train.txt',sep = "\t")
    #dictionary_data.load_txt('data/train_ex.txt')
    all_losses = train_model(Model,train_data,dictionary_data,optimizer,loss_function,params)
    