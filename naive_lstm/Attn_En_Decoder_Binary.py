
# coding: utf-8

# In[1]:

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

MAX_LENGTH = 865
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        #self.embedding = nn.Embedding(input_size, hidden_size)
        self.embedding = nn.Linear(self.input_size, self.hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
    
    def forward(self, input, hidden):
        output = self.embedding(input.float()).view(1, 1,-1)
        output, hidden = self.gru(output, hidden)
        return output, hidden
    
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    
    
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        
       # self.embedding = nn.Embedding(input_size, hidden_size)
        self.embedding = nn.Linear(self.input_size, self.hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input, hidden):
        output = self.embedding(input.float()).view(1,1,-1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        output = self.softmax(output)
        return output, hidden
    
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    
    
class AttentionDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        
        #self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.embedding = nn.Linear(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        
    def forward(self, input, hidden, encoder_output):
        #print("input:", input)
        input = target_transform(input)
        embedded = self.embedding(input.float()).view(1,1,-1)
        embedded = self.dropout(embedded)
        
        attn_weight = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        #print(attn_weight.unsqueeze(0).shape, encoder_output.unsqueeze(0).shape)
        attn_applied = torch.bmm(attn_weight.unsqueeze(0), encoder_output.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        #output = F.log_softmax(self.out(output[0]), dim=1)
        output = F.softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weight
        
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

    
print(torch.cuda.is_available())



""" 
NOTE: 
Encoder RNN input of size (Sentence_length * input_feature)
Encoder RNN output of size (1 * 1 * hidden_size)  should be (num_sentences * 1 * hidden_size)

Decoder RNN input of size 0 (scalar value)
Decoder RNN output of size (1 * target_num)   should be (num_sentences * target_num)
"""


# In[2]:

import pickle

# load data from file
with open("pitch_data.pkl", "rb") as f:
    dic = pickle.load(f)
    train_X = dic["X"]
    train_Y = dic["Y"]
    time_X = dic["time"]
    
for i in range(train_Y.shape[0]):
    train_Y[i] = torch.from_numpy((train_Y[i] == 4).astype(int)).float()


# In[3]:

import random
teacher_forcing_ratio = 1
EOS_token = 2
SOS_token = 3


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, 
          decoder_optimizer, criterion, max_length= MAX_LENGTH, verbose = False):
    encoder_hidden = encoder.init_hidden()
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    
    loss = 0
    
    
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]
        """
        if ei == 0:
            print("input shape:", input_tensor.shape)
            print("output shape:", encoder_output.shape)
        """
        
    decoder_input = torch.Tensor([[0]], device=device)
    
    decoder_hidden = encoder_hidden
    
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    
    temp = []
    
    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            if verbose:
                temp.append(int(torch.argmax(decoder_output, dim = 1).cpu().numpy()))
            loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
            #print(loss)
            decoder_input = target_tensor[di]
            """
            if di == 0:
                print("decoder input shape:", decoder_input.shape)
                print("decoder output shape:", decoder_output.shape)
            """
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, attention = decoder(decoder_input, 
                                                                decoder_hidden, encoder_outputs)  
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
            if decoder_input.item() == EOS_token:
                break

    loss.backward()
    if verbose:
        print("Prediction :", temp) 
        print("Target:", target_tensor) 
    

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


# In[4]:

def input_transform(train_x, time_x, i):
    output = torch.from_numpy(np.array([train_x[i], time_x[i]]))
    return output.transpose(1, 0).to(device)

"""
def target_transform(train_y):
    output = torch.zeros((train_y.shape[0], 7))
    for i in range(train_y.shape[0]):
        output[i, int(train_y[i])] = 1
    return output.unsqueeze(1)
"""


def target_transform(train_y):
    output = torch.zeros((1, 2))
    output[0, int(train_y)] = 1
    return output.unsqueeze(1).to(device)


# In[5]:

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


# In[6]:

def trainIters(encoder, decoder, n_iters, print_every = 1000, plot_every = 100, learning_rate = 0.01, CEL_weight=[1,5]):    
    start = time.time()
    
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0
    
    encoder_optimizer = optim.SGD(encoder.parameters(), lr = learning_rate, weight_decay = 0.95)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr = learning_rate, weight_decay = 0.95)
    
    criterion = nn.CrossEntropyLoss(weight = torch.Tensor(CEL_weight).to(device))
    
    for iter in range(1, n_iters + 1):
        num = iter % 1373
        verbose = (iter % print_every == 0)
        input_tensor = input_transform(train_X, time_X, num - 1).to(device)
        #target_tensor = target_transform(train_Y[num]).long()
        target_tensor = torch.Tensor(train_Y[num]).long().to(device)
        #target_tensor[-1] = EOS_token
        
        loss = train(input_tensor, target_tensor, encoder, decoder, 
                     encoder_optimizer, decoder_optimizer, criterion, verbose = verbose)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

#    showPlot(plot_losses)


# In[ ]:

import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


input_size = 2
hidden_size = 256
output_size = 2

encoder = EncoderRNN(input_size, hidden_size).to(device)
decoder = AttentionDecoder(hidden_size, output_size).to(device)

trainIters(encoder, decoder, 10000, print_every=1000, learning_rate=1e-3, CEL_weight = [0.1,0.9])
    


# In[ ]:



