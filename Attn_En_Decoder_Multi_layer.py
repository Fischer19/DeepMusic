
# coding: utf-8

# In[138]:


import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

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
        
        self.lstm_1 = nn.LSTM(self.output_size, self.hidden_size)
        self.lstm_2 = nn.LSTM(self.output_size, self.hidden_size)
        self.lstm_3 = nn.LSTM(self.output_size, self.hidden_size)
        self.lstm_4 = nn.LSTM(self.output_size, self.hidden_size)
        self.lstm_5 = nn.LSTM(self.output_size, self.hidden_size)
        
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input, ch_1, ch_2, ch_3, ch_4, ch_5):
        (hidden_1, cell_1), (hidden_2, cell_2), (hidden_3, cell_3), (hidden_4, cell_4), (hidden_5, cell_5) = ch_1, ch_2, ch_3, ch_4, ch_5
    
        output, (hidden_1, cell_1) = self.lstm_1(input.view(1,1,-1).float(), (hidden_1, cell_1))
        output = self.out(output)
        output_1 = output
        output, (hidden_2, cell_2) = self.lstm_2(output, (hidden_2, cell_2))
        output = self.out(output)
        output_2 = output
        output, (hidden_3, cell_3) = self.lstm_3(output + output_1, (hidden_3, cell_3)) # skip_connection 1
        output = self.out(output)
        output_3 = output
        output, (hidden_4, cell_4) = self.lstm_4(output + output_2, (hidden_4, cell_4)) # skip_connection 2
        output = self.out(output)
        output_4 = output
        output, (hidden_5, cell_5) = self.lstm_5(output + output_3, (hidden_5, cell_5)) # skip_connection 3
        output = self.out(output[0])
        output = self.softmax(output)
        return output, (hidden_1, cell_1), (hidden_2, cell_2), (hidden_3, cell_3),  (hidden_4, cell_4), (hidden_5, cell_5)
    
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    
    def init_cell(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    
    
print(torch.cuda.is_available())



""" 
NOTE: 
Encoder RNN input of size (Sentence_length * input_feature)
Encoder RNN output of size (1 * 1 * hidden_size)  should be (num_sentences * 1 * hidden_size)

Decoder RNN input of size 0 (scalar value)
Decoder RNN output of size (1 * target_num)   should be (num_sentences * target_num)
"""


# In[180]:


import pickle

# load data from file
with open("pitch_data.pkl", "rb") as f:
    dic = pickle.load(f)
    train_X = dic["X"]
    train_Y = dic["Y"]
    time_X = dic["time"]
    
for i in range(train_Y.shape[0]):
    train_Y[i] = torch.from_numpy((train_Y[i] == 4).astype(int)).float()
    


# In[181]:


def input_transform(train_x, time_x, i):
    output = torch.from_numpy(np.array([train_x[i], time_x[i]]))
    return output.transpose(1, 0).to(device)

def input_factorize(train_x):
    output = []
    for i in range(train_x.shape[0]):
        for item in np.array_split(train_x[i], train_x[i].shape[0] / 9):
            output.append(item)
    return output


def target_factorize(train_y):
    output = []
    for i in range(train_y.shape[0]):
        for item in np.array_split(train_y[i].numpy(), train_y[i].shape[0] / 9):
            output.append(torch.Tensor(item))
    return output

def target_transform(train_y):
    output = torch.zeros((1, 2))
    output[0, int(train_y)] = 1
    return output.unsqueeze(1).to(device)



train_X = input_factorize(train_X)
time_X = input_factorize(time_X)
target_Tensor = target_factorize(train_Y)


# In[92]:


print(len(train_X))
print(target_Tensor[50000])


# In[176]:


import random
teacher_forcing_ratio = 1


def train(input_tensor, target_tensor, decoder, decoder_optimizer, criterion, verbose = False):
    decoder_optimizer.zero_grad()
    
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    
    loss = 0
        
    hidden_1 = decoder.init_hidden()
    hidden_2 = decoder.init_hidden()
    hidden_3 = decoder.init_hidden()
    hidden_4 = decoder.init_hidden()
    hidden_5 = decoder.init_hidden()
    cell_1 = decoder.init_cell()
    cell_2 = decoder.init_cell()
    cell_3 = decoder.init_cell()
    cell_4 = decoder.init_cell()
    cell_5 = decoder.init_cell()
    
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    
    temp = []
    
    decoder_input = input_tensor[0]
    
    if use_teacher_forcing:
        for di in range(1, target_length):
            decoder_output, (hidden_1, cell_1), (hidden_2, cell_2), (hidden_3, cell_3),  (hidden_4, cell_4), (hidden_5, cell_5) = decoder(decoder_input, 
                            (hidden_1, cell_1), (hidden_2, cell_2), (hidden_3, cell_3),  (hidden_4, cell_4), (hidden_5, cell_5))
            if verbose:
                temp.append(int(torch.argmax(decoder_output, dim = 1).cpu().numpy()))
            #print(di, input_tensor.shape)
            #print(input_tensor[di])
            loss += criterion(decoder_output, target_tensor[di].unsqueeze(0).long())
            decoder_input = input_tensor[di]
            """
            if di == 0:
                print("decoder input shape:", decoder_input.shape)
                print("decoder output shape:", decoder_output.shape)
            """
    else:
        for di in range(1, input_length):
            decoder_output, (hidden_1, cell_1), (hidden_2, cell_2), (hidden_3, cell_3),  (hidden_4, cell_4), (hidden_5, cell_5) = decoder(decoder_input, 
                            (hidden_1, cell_1), (hidden_2, cell_2), (hidden_3, cell_3),  (hidden_4, cell_4), (hidden_5, cell_5))
            if verbose:
                temp.append(int(torch.argmax(decoder_output, dim = 1).cpu().numpy()))
            loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
            
            #print(loss)
            decoder_input = decoder_output

    loss.backward()
    if verbose:
        print("Prediction :", temp) 
        print("Target:", target_tensor) 
    

    decoder_optimizer.step()

    return loss.item() / target_length


# In[153]:


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


# In[182]:


def trainIters(decoder, n_iters, print_every = 1000, plot_every = 100, learning_rate = 0.01, CEL_weight=[1,5]):    
    start = time.time()
    
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0
    
    decoder_optimizer = optim.SGD(decoder.parameters(), lr = learning_rate, weight_decay = 0.95)
    
    criterion = nn.CrossEntropyLoss(weight = torch.Tensor(CEL_weight).to(device))
    
    for iter in range(1, n_iters + 1):
        num = iter % 51954
        verbose = (iter % print_every == 0)
        input_tensor = input_transform(train_X, time_X, num - 1).to(device)
        target_tensor = target_Tensor[num].to(device)
        #print(input_tensor.shape)
        #print(target_tensor.shape)
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
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


# In[184]:


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

decoder = DecoderRNN(hidden_size, output_size).to(device)

trainIters(decoder, 10000, print_every=100, learning_rate=1e-3, CEL_weight = [0.1,0.9])
    


# In[174]:




