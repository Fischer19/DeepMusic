
# coding: utf-8

# In[22]:

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.lstm_1 = nn.LSTM(self.input_size, self.hidden_size)
        self.lstm_2 = nn.LSTM(self.hidden_size, self.hidden_size)
        self.lstm_3 = nn.LSTM(self.hidden_size, self.hidden_size)
        self.lstm_4 = nn.LSTM(self.hidden_size, self.hidden_size)
        self.lstm_5 = nn.LSTM(self.hidden_size, self.hidden_size)
        self.lstm_6 = nn.LSTM(self.hidden_size, self.hidden_size)
        self.lstm_7 = nn.LSTM(self.hidden_size, self.hidden_size)
        
        self.out = nn.Linear(self.hidden_size, self.output_size)
          
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input, ch_1, ch_2, ch_3, ch_4, ch_5, ch_6, ch_7):
        (hidden_1, cell_1), (hidden_2, cell_2), (hidden_3, cell_3), (hidden_4, cell_4), (hidden_5, cell_5) = ch_1, ch_2, ch_3, ch_4, ch_5
        (hidden_6, cell_6), (hidden_7, cell_7) = ch_6, ch_7
        
        
        output, (hidden_1, cell_1) = self.lstm_1(input.view(1,1,-1).float(), (hidden_1, cell_1))
        output_1 = output
        
        output, (hidden_2, cell_2) = self.lstm_2(output, (hidden_2, cell_2))
        output_2 = output
        
        output, (hidden_3, cell_3) = self.lstm_3(output + output_1, (hidden_3, cell_3)) # skip_connection 1
        output_3 = output
        
        output, (hidden_4, cell_4) = self.lstm_4(output + output_2, (hidden_4, cell_4)) # skip_connection 2
        output_4 = output
        
        output, (hidden_5, cell_5) = self.lstm_5(output + output_3, (hidden_5, cell_5)) # skip_connection 3
        output_5 = output
        
        output, (hidden_6, cell_6) = self.lstm_6(output + output_4, (hidden_6, cell_6)) # skip_connection 4
        
        output, (hidden_7, cell_7) = self.lstm_7(output + output_5, (hidden_7, cell_7)) # skip_connection 5
        
        output = self.out(output[0])
        output = self.softmax(output)
        return output, (hidden_1, cell_1),(hidden_2, cell_2),(hidden_3, cell_3),(hidden_4, cell_4),(hidden_5, cell_5),(hidden_6, cell_6),(hidden_7, cell_7)
    
    def init_hidden(self):
        return torch.rand((1, 1, self.hidden_size), device=device)/100
    
    def init_cell(self):
        return torch.rand((1, 1, self.hidden_size), device=device)/100
    
    
print(torch.cuda.is_available())



""" 
NOTE: 
Encoder RNN input of size (Sentence_length * input_feature)
Encoder RNN output of size (1 * 1 * hidden_size)  should be (num_sentences * 1 * hidden_size)

Decoder RNN input of size 0 (scalar value)
Decoder RNN output of size (1 * target_num)   should be (num_sentences * target_num)
"""


# In[5]:

import pickle

# load data from file
"""with open("/home/yiqin/2018summer_project/DeepMusic/pitch_data.pkl", "rb") as f:
    dic = pickle.load(f)
    train_X = dic["X"]
    #train_Y = dic["Y"]
    #time_X = dic["time"]
    """
    
with open("/home/yiqin/2018summer_project/balanced_data_small.pkl", "rb") as f:
    dic = pickle.load(f)
    train_X = dic["X2"]
    train_Y = dic["Y"]
    
    
target_Tensor = train_Y
maximum_target = len(train_Y)

def focal_loss(gamma, rescale, criterion, output, target):
    if int(target) == 1:
        p_negative = (1 - output[0,1])**gamma
        loss = rescale * p_negative * criterion(output, target.unsqueeze(0).long())
    else:
        p_negative = (1 - output[0,0])**gamma
        loss = p_negative * criterion(output, target.unsqueeze(0).long())
    return loss


import random
teacher_forcing_ratio = 1


def train(input_tensor, target_tensor, decoder, decoder_optimizer, criterion, verbose = False):
    decoder_optimizer.zero_grad()
    
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    
    hidden_1 = decoder.init_hidden()
    hidden_2 = decoder.init_hidden()
    hidden_3 = decoder.init_hidden()
    hidden_4 = decoder.init_hidden()
    hidden_5 = decoder.init_hidden()
    hidden_6 = decoder.init_hidden()
    hidden_7 = decoder.init_hidden()
    cell_1 = decoder.init_cell()
    cell_2 = decoder.init_cell()
    cell_3 = decoder.init_cell()
    cell_4 = decoder.init_cell()
    cell_5 = decoder.init_cell()
    cell_6 = decoder.init_cell()
    cell_7 = decoder.init_cell()
    
    loss = 0
    
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    
    temp = []
    temp_score = []
    
    decoder_input = input_tensor[0]
    
    for di in range(0, target_length):
        decoder_output, (hidden_1, cell_1), (hidden_2, cell_2), (hidden_3, cell_3), (hidden_4, cell_4), (hidden_5, cell_5), (hidden_6, cell_6), (hidden_7, cell_7) = decoder(decoder_input, 
                        (hidden_1, cell_1), (hidden_2, cell_2), (hidden_3, cell_3), (hidden_4, cell_4), (hidden_5, cell_5), (hidden_6, cell_6), (hidden_7, cell_7))
        if verbose:
            temp.append(int(torch.argmax(decoder_output, dim = 1).cpu().numpy()))
            temp_score.append(decoder_output)

        loss += criterion(decoder_output, target_tensor[di].unsqueeze(0).long())

        if di + 1 < target_length:
            decoder_input = input_tensor[di + 1]

    loss.backward()

    if verbose:
        print("Prediction :", temp) 
        print("Target:", target_tensor.squeeze())
        print("Score :", temp_score)
        
    

    decoder_optimizer.step()

    return loss.item() / target_length

# In[63]:

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# In[64]:

def trainIters(decoder, n_iters, print_every = 1000, plot_every = 100, learning_rate = 0.01, CEL_weight=[1,5], total_batch = maximum_target, gamma = 0.1):    
    start = time.time()
    
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0
    
    decoder_optimizer = optim.Adagrad(decoder.parameters(), lr = learning_rate)
    
    criterion = nn.CrossEntropyLoss(weight = torch.Tensor(CEL_weight).to(device))
    
    scheduler = optim.lr_scheduler.StepLR(decoder_optimizer, step_size = total_batch, gamma = gamma)
    
    
    for iter in range(1, n_iters + 1):
        num = iter % total_batch
        verbose = (iter % print_every == 0)
        input_tensor = train_X[num].to(device)
        target_tensor = target_Tensor[num].to(device)
        input_tensor = Variable(input_tensor, requires_grad = True)
        #print(input_tensor.shape, target_tensor.shape)
        if input_tensor.shape[0]<2:
            continue
        if input_tensor.shape[0] != target_tensor.shape[0]:
            continue
        
        loss = train(input_tensor, target_tensor, decoder, 
                     decoder_optimizer, criterion, verbose = verbose)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * print_every, print_loss_avg))
            torch.save(decoder.state_dict(), 'naive_lstm_train.pt')
        
        scheduler.step()
        


input_size = 2
hidden_size = 256
output_size = 2

decoder = DecoderRNN(input_size, hidden_size, output_size).to(device)

trainIters(decoder, 300000, print_every=1000, learning_rate=1e-2, gamma=0.1)


    


def evaluate(decoder, test_X):
    input_length = input_tensor.size(0)
    
    loss = 0
        
    hidden_1 = decoder.init_hidden()
    hidden_2 = decoder.init_hidden()
    hidden_3 = decoder.init_hidden()
    hidden_4 = decoder.init_hidden()
    hidden_5 = decoder.init_hidden()
    hidden_6 = decoder.init_hidden()
    hidden_7 = decoder.init_hidden()
    cell_1 = decoder.init_cell()
    cell_2 = decoder.init_cell()
    cell_3 = decoder.init_cell()
    cell_4 = decoder.init_cell()
    cell_5 = decoder.init_cell()
    cell_6 = decoder.init_cell()
    cell_7 = decoder.init_cell()
    
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    
    temp = []
    temp_score = []
    
    decoder_input = input_tensor[0]
    
    for di in range(0, input_length):
        decoder_output, (hidden_1, cell_1), (hidden_2, cell_2), (hidden_3, cell_3),  (hidden_4, cell_4), (hidden_5, cell_5), (hidden_6, cell_6), (hidden_7, cell_7) = decoder(decoder_input, 
                        (hidden_1, cell_1), (hidden_2, cell_2), (hidden_3, cell_3), (hidden_4, cell_4), (hidden_5, cell_5), (hidden_6, cell_6), (hidden_7, cell_7))
        temp.append(int(torch.argmax(decoder_output, dim = 1).cpu().numpy()))
        temp_score.append(decoder_output)
        if di + 1 < input_length:
            decoder_input = input_tensor[di+1]

    return temp


