import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import math
import time
import os
import numpy as np

from NN_Models_Load_Emb import Sim_CNN_RNN,Com_CNN_RNN
from NN_Models_Load_batch_RNN import Simple_RNN,Linear_RNN,Complex_RNN

def Peason_Score(predict_score,target_score):
	print(predict_score)
	print(target_score)
	x = predict_score - predict_score.mean()
	y = target_score - target_score.mean()
	return x.dot(y) / (x.std() * y.std())

def sep_sentences(trian_path,test_path,save_trian,save_test,save_vocab):
	vocab_to_index = {}
	fpW = open(save_trian,"w",encoding = "utf-8")
	fp = open(trian_path,"r",encoding = "utf-8")
	fp.readline()
	while True:
		line = fp.readline()
		if not line:
			break
		line = line.strip().split("	")
		index = line[0]
		sentA = line[1]
		sentB = line[2]
		vocab_list = sentA.split() + sentB.split()
		for word in vocab_list:
			if word not in vocab_to_index:
				vocab_to_index[word] = len(vocab_to_index)

		score = float(line[3])
		fpW.write(sentA + "\n" + sentB + "\n" + str(score) + "\n\n")
	fp.close()
	fpW.close()

	fpW = open(save_test,"w",encoding = "utf-8")
	fp = open(test_path,"r",encoding = "utf-8")
	fp.readline()
	while True:
		line = fp.readline()
		if not line:
			break
		line = line.strip().split("	")
		index = line[0]
		sentA = line[1]
		sentB = line[2]
		vocab_list = sentA.split() + sentB.split()
		for word in vocab_list:
			if word not in vocab_to_index:
				vocab_to_index[word] = len(vocab_to_index)

		score = float(line[3])
		fpW.write(sentA + "\n" + sentB + "\n" + str(score) + "\n\n")
	fp.close()
	fpW.close()

	fpW_vocab = open(save_vocab,"w",encoding = "utf-8")
	for word in vocab_to_index:
		fpW_vocab.write(str(vocab_to_index[word]) + "\t" + word + "\n")
	fpW_vocab.close()
def Get_sentences_pairs(sent_path):
	sentence_pairs = {}
	num = 0
	Temp_Sent = []
	fp = open(sent_path,"r",encoding = "utf-8")
	while True:
		line = fp.readline()
		if not line:
			break
		line = line.strip()
		if line == "":
			sentence_pairs[num] = Temp_Sent
			num += 1
			Temp_Sent = []
		else:
			Temp_Sent.append(line)
			# pass
			# print(line)
	fp.close()
	return sentence_pairs
def Get_vocabs(vocab_path):
	vocab_to_index = {}
	fp = open(vocab_path)
	while True:
		line = fp.readline()
		if not line:
			break
		line = line.strip().split()
		vocab_to_index[line[1]] = int(line[0])
	fp.close()
	return vocab_to_index
def Change_Score(score):
	score = score - 1
	list_q = torch.zeros(2)
	list_q[0] = score / 4
	list_q[1] = (4 - score)/4
	return Variable(list_q)
def seq_to_index(sentence,vocab_to_index):
	ids = [vocab_to_index[w] for w in sentence]
	tensor = torch.LongTensor(ids)
	return Variable(tensor)
def train_iter(Train_Model,optimizer,loss_functions,target_score,sentA,sentB):
	optimizer.zero_grad()
	Train_Model.initialize_hidden_layer()
	predict_score = Train_Model(sentA,sentB)
	loss = loss_functions(predict_score,target_score)
	loss.backward()
	optimizer.step()
	return loss
def time_diff(start,end):
	diff_time_all = start - end
	seconds = diff_time_all % 60
	mintues = math.floor(diff_time_all/60)
	hours = 0
	if mintues !=0:
		hours = math.floor((diff_time_all - mintues*60 -seconds)/60)
	return seconds,mintues,hours
def trian_model(sentence_pairs,Train_Model,optimizer,loss_function,
				epoches,vocab_to_index,model_save_path):
	num = 0
	all_num = 0
	q_t = []
	ex_t = []
	plt.ion()
	for k in range(epoches):
		start = time.clock()
		Length = len(sentence_pairs)
		sum_temp = 0
		for index in sentence_pairs:
			sent_p = sentence_pairs[index]
			sentA = sent_p[0].split()
			sentB = sent_p[1].split()
			score = float(sent_p[2])
			sentA = seq_to_index(sentA,vocab_to_index)
			sentB = seq_to_index(sentB,vocab_to_index)
			target_score = Change_Score(score)
			loss = train_iter(Train_Model,optimizer,loss_function,target_score,sentA,sentB)

			print(loss.data[0])
			ex_t.append(loss.data[0])
			num += 1
			all_num += 1
			sum_temp += loss.data[0]
			if num % 100 == 0:
				q_t.append(sum_temp/num)
				x = np.linspace(0,len(q_t),len(q_t))
				y = np.array(q_t)
				plt.cla()
				plt.plot(x,y)
				plt.title("The data of training loss (Global Loss:%.4f)"%(sum(ex_t)/len(ex_t)),fontdict = {"size":15,"color":"red"})
				plt.xlabel("The number of sentence pairs")
				plt.ylabel("Loss")
				plt.pause(0.1)
				sum_temp = 0
				num = 0
		q_t = []
		end = time.clock()
		seconds,minutes,hours = time_diff(start,end)
		print("time differential:%d hours,%d mintues,%d seconds"%(hours,minutes,seconds))
		if not os.path.exists("pictures"):
			os.mkdir("pictures")
		plt.savefig("pictures/" + str(k) + ".jpg")

	fpW = open("pictures/"+"loss.txt","w",encoding = "utf-8")
	for data in ex_t:
		fpW.write(str(data) + "\n")
	fpW.close()
	torch.save(Train_Model,model_save_path)
def test_sentence_pairs(test_path,vocab_to_index,model_path):
	test_sentence_pairs = Get_sentences_pairs(test_path)
	data_real_score = []
	data_predict_score = []
	model = torch.load(model_path)
	num = 0
	for k in test_sentence_pairs:
		num += 1
		if num%100 == 0:
			print("processed sentence:%d"%num)
		sentA = test_sentence_pairs[k][0].split()
		sentB = test_sentence_pairs[k][1].split()
		sentA = seq_to_index(sentA,vocab_to_index)
		sentB = seq_to_index(sentB,vocab_to_index)
		score_temp = model(sentA,sentB)
		#print(score_temp.data.numpy()[0][0]*5)
		data_predict_score.append(score_temp.data.numpy()[0][0]*4)
		data_real_score.append(float(test_sentence_pairs[k][2]))
	data_real_score = np.array(data_real_score)
	data_predict_score = np.array(data_predict_score)
	score = Peason_Score(data_predict_score,data_real_score)
	return score
def train_Simple_RNN(trian_path,test_path,save_trian,save_test,save_vocab,path_file):
	trian_path = "sentences/SICK_train.txt"
	test_path = "sentences/SICK_test_annotated.txt"
	save_trian = "Sent_B/SICK_train.txt"
	save_test = "Sent_B/SICK_test.txt"
	save_vocab = "Sent_B/vocab.txt"

	path_file = "Sent_B"
	if not os.path.exists(path_file):
		os.mkdir(path_file)
	if not os.path.exists(save_trian) or os.path.exists(save_test):
		sep_sentences(trian_path,test_path,save_trian,save_test,save_vocab)
	sentence_pairs = Get_sentences_pairs(save_trian)
	vocab_to_index =Get_vocabs(save_vocab)

	vocab_size = len(vocab_to_index)
	embedding_dim = 400
	hidden_dim = 250
	temp_dim = 20
	out_dim = 2
	algorithm_q = "LSTM"
	rnn = Simple_RNN(vocab_size,embedding_dim,hidden_dim,temp_dim,out_dim,dp = 0.2,algorithm = algorithm_q,bi = True)
	# RNN Model + RMSprop:lr=1e-4,alpha=0.9, eps=1e-08, weight_decay=0, momentum=0
	# RNN Model + opt_Rprop lr=LR, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
	# optimizer = torch.optim.Rprop(rnn.parameters(), lr=1e-4, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
	optimizer = optim.RMSprop(rnn.parameters(), lr=1e-4,
					alpha=0.1, eps=1e-08, weight_decay=0.9, momentum=0.8, centered=False)
	loss_function = nn.BCELoss()

	epoches = 1
	model_save_path = "Simple_RNN_"+rnn.algorithm + ".model"
	if not os.path.exists(model_save_path):
		trian_model(sentence_pairs,rnn,optimizer,loss_function,
				epoches,vocab_to_index,model_save_path)
	score = test_sentence_pairs(save_test,vocab_to_index,model_save_path)
	print("Pearson score:%.4f"%score)
def train_Linear_RNN(trian_path,test_path,save_trian,save_test,save_vocab,path_file):
	trian_path = "sentences/SICK_train.txt"
	test_path = "sentences/SICK_test_annotated.txt"
	save_trian = "Sent_B/SICK_train.txt"
	save_test = "Sent_B/SICK_test.txt"
	save_vocab = "Sent_B/vocab.txt"

	path_file = "Sent_B"
	if not os.path.exists(path_file):
		os.mkdir(path_file)
	if not os.path.exists(save_trian) or os.path.exists(save_test):
		sep_sentences(trian_path,test_path,save_trian,save_test,save_vocab)
	sentence_pairs = Get_sentences_pairs(save_trian)
	vocab_to_index =Get_vocabs(save_vocab)

	vocab_size = len(vocab_to_index)
	embedding_dim = 400
	hidden_dim = 250
	out_dim = 2
	algorithm_q = "RNN"

	rnn = Linear_RNN(vocab_size,embedding_dim,hidden_dim,out_dim,
				dp = 0.2,num_layers = 10,
				algorithm = algorithm_q,
				bi = True)
	optimizer = optim.RMSprop(rnn.parameters(), lr=1e-4,
					alpha=0.9, eps=1e-08, weight_decay=0, momentum=0.8, centered=False)
	loss_function = nn.BCELoss()

	epoches = 5
	model_save_path = "Linear_RNN_"+rnn.algorithm + ".model"
	trian_model(sentence_pairs,rnn,optimizer,loss_function,
				epoches,vocab_to_index,model_save_path)
	test_sentence_pairs(save_test,vocab_to_index,model_save_path)
def train_Complex_RNN(trian_path,test_path,save_trian,save_test,save_vocab,path_file):
	trian_path = "sentences/SICK_train.txt"
	test_path = "sentences/SICK_test_annotated.txt"
	save_trian = "Sent_B/SICK_train.txt"
	save_test = "Sent_B/SICK_test.txt"
	save_vocab = "Sent_B/vocab.txt"

	path_file = "Sent_B"
	if not os.path.exists(path_file):
		os.mkdir(path_file)
	if not os.path.exists(save_trian) or os.path.exists(save_test):
		sep_sentences(trian_path,test_path,save_trian,save_test,save_vocab)
	sentence_pairs = Get_sentences_pairs(save_trian)
	vocab_to_index =Get_vocabs(save_vocab)

	vocab_size = len(vocab_to_index)
	embedding_dim = 400
	hidden_dim = 250
	out_dim = 2
	algorithm_q = "GRU"

	rnn = Complex_RNN(vocab_size,embedding_dim,hidden_dim,out_dim,
				dp = 0.2,num_layers = 10,
				algorithm = algorithm_q,
				bi = True)

	optimizer = optim.RMSprop(rnn.parameters(), lr=1e-4,
					alpha=0.9, eps=1e-08, weight_decay=0, momentum=0.8, centered=False)
	loss_function = nn.BCELoss()

	epoches = 5
	model_save_path = "Complex_RNN_"+rnn.algorithm + ".model"
	if not os.path.exists(model_save_path):
		trian_model(sentence_pairs,rnn,optimizer,loss_function,
				epoches,vocab_to_index,model_save_path)
	test_sentence_pairs(save_test,vocab_to_index,model_save_path)
def train_Sim_CNN_RNN(trian_path,test_path,save_trian,save_test,save_vocab,path_file):
	trian_path = "sentences/SICK_train.txt"
	test_path = "sentences/SICK_test_annotated.txt"
	save_trian = "Sent_B/SICK_train.txt"
	save_test = "Sent_B/SICK_test.txt"
	save_vocab = "Sent_B/vocab.txt"

	path_file = "Sent_B"
	if not os.path.exists(path_file):
		os.mkdir(path_file)
	if not os.path.exists(save_trian) or os.path.exists(save_test):
		sep_sentences(trian_path,test_path,save_trian,save_test,save_vocab)
	sentence_pairs = Get_sentences_pairs(save_trian)
	vocab_to_index =Get_vocabs(save_vocab)

	vocab_size = len(vocab_to_index)
	embedding_dim = 400
	hidden_dim = 250
	out_dim = 2
	algorithm_q = "GRU"

	rnn = Sim_CNN_RNN(vocab_size,embedding_dim,hidden_dim,out_dim,num_layers_n = 80,
				kernel_size = 3,stride = 1,padding = 1,dilation = 1,algorithm = algorithm_q)
	optimizer = optim.RMSprop(rnn.parameters(), lr=1e-4,
					alpha=0.9, eps=1e-08, weight_decay=0, momentum=0.8, centered=False)
	loss_function = nn.BCELoss()

	epoches = 5
	model_save_path = "Sim_CNN_RNN_"+rnn.algorithm + ".model"
	trian_model(sentence_pairs,rnn,optimizer,loss_function,
				epoches,vocab_to_index,model_save_path)
	test_sentence_pairs(save_test,model_save_path)
def train_Com_CNN_RNN(trian_path,test_path,save_trian,save_test,save_vocab,path_file):
	trian_path = "sentences/SICK_train.txt"
	test_path = "sentences/SICK_test_annotated.txt"
	save_trian = "Sent_B/SICK_train.txt"
	save_test = "Sent_B/SICK_test.txt"
	save_vocab = "Sent_B/vocab.txt"

	path_file = "Sent_B"
	if not os.path.exists(path_file):
		os.mkdir(path_file)
	if not os.path.exists(save_trian) or os.path.exists(save_test):
		sep_sentences(trian_path,test_path,save_trian,save_test,save_vocab)
	sentence_pairs = Get_sentences_pairs(save_trian)
	vocab_to_index =Get_vocabs(save_vocab)

	vocab_size = len(vocab_to_index)
	embedding_dim = 400
	hidden_dim = 250
	out_dim = 2
	algorithm_qA = "RNN"
	algorithm_qB = "RNN"

	rnn = Com_CNN_RNN(vocab_size,embedding_dim,hidden_dim,temp_dim = 50,out_dim = 2,num_layers = 80,
				num_epochs = 2,algorithmA = algorithm_qA,algorithmB = algorithm_qB)
	optimizer = optim.RMSprop(rnn.parameters(), lr=1e-4,
					alpha=0.9, eps=1e-08, weight_decay=0, momentum=0.8, centered=False)
	loss_function = nn.BCELoss()

	epoches = 5
	model_save_path = "Sim_CNN_RNN_"+rnn.algorithmA + "-"+rnn.algorithmB+ ".model"
	trian_model(sentence_pairs,rnn,optimizer,loss_function,
				epoches,vocab_to_index,model_save_path)
	test_sentence_pairs(save_test,model_save_path)
def Main():
	trian_path = "sentences/SICK_train.txt"
	test_path = "sentences/SICK_test_annotated.txt"
	save_trian = "Sent_B/SICK_train.txt"
	save_test = "Sent_B/SICK_test.txt"
	save_vocab = "Sent_B/vocab.txt"

	path_file = "Sent_B"
	train_Complex_RNN(trian_path,test_path,save_trian,save_test,save_vocab,path_file)
if __name__ == "__main__":
	Main()
