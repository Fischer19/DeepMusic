
# coding: utf-8

# In[82]:

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
    
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.lstm_1 = nn.LSTM(self.output_size, self.hidden_size)
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


# In[36]:

import pickle

# load data from file
with open("pitch_data.pkl", "rb") as f:
    dic = pickle.load(f)
    train_X = dic["X"]
    train_Y = dic["Y"]
    time_X = dic["time"]
    
for i in range(train_Y.shape[0]):
    train_Y[i] = torch.from_numpy((train_Y[i] == 4).astype(int)).float()
    


# In[37]:

def input_transform(train_x, time_x, i):
    output = torch.from_numpy(np.array([train_x[i], time_x[i]]))
    return output.transpose(1, 0).to(device)

def input_factorize(train_x):
    output = []
    for i in range(train_x.shape[0]):
        for item in np.array_split(train_x[i], train_x[i].shape[0] / 7):
            output.append(item)
    return output


def target_factorize(train_y):
    output = []
    for i in range(train_y.shape[0]):
        for item in np.array_split(train_y[i].numpy(), train_y[i].shape[0] / 7):
            output.append(torch.Tensor(item))
    return output

def target_transform(train_y):
    output = torch.zeros((1, 2))
    output[0, int(train_y)] = 1
    return output.unsqueeze(1).to(device)



train_X = input_factorize(train_X)
time_X = input_factorize(time_X)
target_Tensor = target_factorize(train_Y)


# In[38]:

print(len(train_X))
print(target_Tensor[50000])


# In[68]:

import random
teacher_forcing_ratio = 1


def train(input_tensor, target_tensor, decoder, decoder_optimizer, criterion, verbose = False):
    decoder_optimizer.zero_grad()
    
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    
    loss = 0
    
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    
    temp = []
    temp_score = []
    
    decoder_input = input_tensor[0]
    
    if use_teacher_forcing:
        for di in range(0, target_length):
            decoder_output, (hidden_1, cell_1), (hidden_2, cell_2), (hidden_3, cell_3), (hidden_4, cell_4), (hidden_5, cell_5), (hidden_6, cell_6), (hidden_7, cell_7) = decoder(decoder_input, 
                            (hidden_1, cell_1), (hidden_2, cell_2), (hidden_3, cell_3), (hidden_4, cell_4), (hidden_5, cell_5), (hidden_6, cell_6), (hidden_7, cell_7))
            if verbose:
                temp.append(int(torch.argmax(decoder_output, dim = 1).cpu().numpy()))
                temp_score.append(decoder_output)
                #print("decoder_output:", decoder_output)
                
            #print(input_tensor[di])
            if int(target_tensor[di]) == 1:
                #temp = float(loss)
                loss += 5 * criterion(decoder_output, target_tensor[di].unsqueeze(0).long())
                #print(decoder_output, target_tensor[di])
                #print(float(loss) - temp)
            else:
                #temp = float(loss)
                loss += criterion(decoder_output, target_tensor[di].unsqueeze(0).long())
                #print(decoder_output, target_tensor[di])
                #print(float(loss) - temp)
            if di + 1 < target_length:
                decoder_input = input_tensor[di + 1]
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
    """if verbose:
        for name, item in decoder.named_parameters():
            if item.requires_grad:
                print(name)
                print(item.shape)
                print(item.grad)
"""
    if verbose:
        print("Prediction :", temp) 
        print("Score :", temp_score)
        print("Target:", target_tensor) 
    

    decoder_optimizer.step()

    return loss.item() / target_length


# In[69]:

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


# In[70]:

def trainIters(decoder, n_iters, print_every = 1000, plot_every = 100, learning_rate = 0.01, CEL_weight=[1,5], total_batch = 51954):    
    start = time.time()
    
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0
    
    decoder_optimizer = optim.Adagrad(decoder.parameters(), lr = learning_rate)
    
    criterion = nn.CrossEntropyLoss(weight = torch.Tensor(CEL_weight).to(device))
    
    
    for iter in range(1, n_iters + 1):
        num = iter % total_batch
        verbose = (iter % print_every == 0)
        input_tensor = input_transform(train_X, time_X, num - 1).to(device)
        target_tensor = target_Tensor[num].to(device)
        input_tensor = Variable(input_tensor, requires_grad = True)
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


# In[71]:

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

trainIters(decoder, 10000, print_every=100, learning_rate=1e-3, CEL_weight = [1,10], total_batch=5)
    


# In[ ]:




# In[55]:

print(decoder.lstm_1.state_dict().keys())
print(decoder.lstm_1.state_dict()['weight_ih_l0'].grad)


# In[77]:

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


# In[81]:

for i in range(5):
    target_tensor = target_Tensor[i].to(device)
    input_tensor = input_transform(train_X, time_X, i-1).to(device)
    print(evaluate(decoder, input_tensor))
    print(target_tensor)


# In[ ]:



