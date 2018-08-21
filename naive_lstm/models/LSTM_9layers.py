
# coding: utf-8

# naive LSTM model trained with smooth data

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from torch.autograd import Variable

device = torch.device(3 if torch.cuda.is_available() else "cpu")
    
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
        self.lstm_8 = nn.LSTM(self.hidden_size, self.hidden_size)
        self.lstm_9 = nn.LSTM(self.hidden_size, self.hidden_size)
        
        
        self.out = nn.Linear(self.hidden_size, self.output_size)
          
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input, ch_1, ch_2, ch_3, ch_4, ch_5, ch_6, ch_7, ch_8, ch_9):
        (hidden_1, cell_1), (hidden_2, cell_2), (hidden_3, cell_3), (hidden_4, cell_4), (hidden_5, cell_5) = ch_1, ch_2, ch_3, ch_4, ch_5
        (hidden_6, cell_6), (hidden_7, cell_7) = ch_6, ch_7
        (hidden_8, cell_8), (hidden_9, cell_9) = ch_8, ch_9
        
        
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
        output_6 = output
        output, (hidden_7, cell_7) = self.lstm_7(output + output_5, (hidden_7, cell_7)) # skip_connection 5
        output_7 = output
        
        output, (hidden_8, cell_8) = self.lstm_8(output + output_6, (hidden_8, cell_8)) # skip_connection 6
        
        output, (hidden_9, cell_9) = self.lstm_9(output + output_7, (hidden_9, cell_9)) # skip_connection 7
        
        
        output = self.out(output[0])
        #output = self.softmax(output)
        return output, (hidden_1, cell_1),(hidden_2, cell_2),(hidden_3, cell_3),(hidden_4, cell_4),(hidden_5, cell_5),(hidden_6, cell_6),(hidden_7, cell_7),(hidden_8, cell_8),(hidden_9, cell_9)
    
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
    
with open("/home/yiqin/2018summer_project/data/smooth_data_augmented.pkl", "rb") as f:
    dic = pickle.load(f)
    train_X = dic["X"]
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

def penalty_loss(penalty, criterion, output, target):
    if int(target) == 1:
        loss = penalty[0] * criterion(output, target)
    else:
        loss = penalty[1] * criterion(output, target)
    return loss



def train(input_tensor, target_tensor, decoder, decoder_optimizer, criterion, verbose = False, penalty = (1,0.5)):
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
    hidden_8 = decoder.init_hidden()
    hidden_9 = decoder.init_hidden()
    cell_1 = decoder.init_cell()
    cell_2 = decoder.init_cell()
    cell_3 = decoder.init_cell()
    cell_4 = decoder.init_cell()
    cell_5 = decoder.init_cell()
    cell_6 = decoder.init_cell()
    cell_7 = decoder.init_cell()
    cell_8 = decoder.init_cell()
    cell_9 = decoder.init_cell()
    
    loss = 0
    

    temp = []
    temp_score = []
    
    decoder_input = input_tensor[0]
    
    for di in range(0, target_length):
        decoder_output, (hidden_1, cell_1), (hidden_2, cell_2), (hidden_3, cell_3), (hidden_4, cell_4), (hidden_5, cell_5), (hidden_6, cell_6), (hidden_7, cell_7),(hidden_8, cell_8), (hidden_9, cell_9) = decoder(decoder_input, 
                        (hidden_1, cell_1), (hidden_2, cell_2), (hidden_3, cell_3), (hidden_4, cell_4), (hidden_5, cell_5), (hidden_6, cell_6), (hidden_7, cell_7), (hidden_8, cell_8), (hidden_9, cell_9))
        if verbose:
            output = float(decoder_output.data.cpu().numpy())
            temp.append(str('%.4f'%output))
            #temp_score.append(decoder_output)

        loss += penalty_loss(penalty, criterion, decoder_output.squeeze(0), target_tensor[di].float())
        
        if di + 1 < target_length:
            decoder_input = input_tensor[di + 1]

    loss.backward()

    if verbose:
        print("Prediction :", temp) 
        print("Target:", target_tensor.squeeze())
        #print("Score :", temp_score)
        
    

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

def trainIters(decoder, n_iters, print_every = 1000, plot_every = 100, learning_rate = 0.01, total_batch = maximum_target, penalty = (1, 0.5), gamma = 0.1):    
    start = time.time()
    
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0
    
    decoder_optimizer = optim.SGD(decoder.parameters(), lr = learning_rate)
    
    criterion = nn.MSELoss()
    
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
                     decoder_optimizer, criterion, verbose = verbose, penalty = penalty)
        print_loss_total += loss

        if iter % print_every == 0:
            acc, one_acc = validate(decoder, train_X[15000:16000], train_Y[15000:16000])
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            print("validation accuracy:", acc)
            print("validation prediction accuracy:", one_acc)
            torch.save(decoder.state_dict(), 'lstm_9l_smooth_train{}.pt'.format(iter/total_batch))
        
        scheduler.step()
        
def validate(decoder, val_x, val_y, val_threshold = 0.5):

    hidden_1 = decoder.init_hidden()
    hidden_2 = decoder.init_hidden()
    hidden_3 = decoder.init_hidden()
    hidden_4 = decoder.init_hidden()
    hidden_5 = decoder.init_hidden()
    hidden_6 = decoder.init_hidden()
    hidden_7 = decoder.init_hidden()
    hidden_8 = decoder.init_hidden()
    hidden_9 = decoder.init_hidden()
    cell_1 = decoder.init_cell()
    cell_2 = decoder.init_cell()
    cell_3 = decoder.init_cell()
    cell_4 = decoder.init_cell()
    cell_5 = decoder.init_cell()
    cell_6 = decoder.init_cell()
    cell_7 = decoder.init_cell()
    cell_8 = decoder.init_cell()
    cell_9 = decoder.init_cell()
    
    temp = []
    temp_score = []
    count = 0
    total = 0
    total_1 = 0
    for i in range(len(val_x)):
        X = val_x[i].to(device).float()
        val_input = X[0]
        input_length = X.size(0)
        result = []
        for di in range(0, input_length):
            decoder_output, (hidden_1, cell_1), (hidden_2, cell_2), (hidden_3, cell_3),  (hidden_4, cell_4), (hidden_5, cell_5), (hidden_6, cell_6), (hidden_7, cell_7), (hidden_8, cell_8), (hidden_9, cell_9) = decoder(val_input, 
                            (hidden_1, cell_1), (hidden_2, cell_2), (hidden_3, cell_3), (hidden_4, cell_4), (hidden_5, cell_5), (hidden_6, cell_6), (hidden_7, cell_7), (hidden_8, cell_8), (hidden_9, cell_9))
            output = float(decoder_output.data.cpu().numpy())
            score = float(decoder_output.item())
            result.append(score)
            if di + 1 < input_length:
                val_input = X[di+1]
        for di, item in enumerate(result):
            if di == (len(result) - 1) and int(item > val_threshold):
                count += 1.0
                total += 1.0
                total_1 += 1.0
                continue
            elif di == 0 :
                continue
            if int(val_y[i][di].item()) == 1:
                total += 1.0
            if  int(item > val_threshold) and (item > max(result[di-1], result[di+1])):
                total_1 += 1.0
                if int(val_y[i][di].item()):
                    count += 1.0

    acc = str('%.4f'%((count / total * 100)) + "%")
    if total_1 == 0.0:
        one_acc = str('%.4f'%(0) + "%")
    else:
        one_acc = str('%.4f'%((count / total_1 * 100)) + "%")
    return acc, one_acc

        


input_size = 2
hidden_size = 256
output_size = 1

decoder = DecoderRNN(input_size, hidden_size, output_size).to(device)

trainIters(decoder, 100000, print_every=1000, learning_rate=1e-2, total_batch = 15000, penalty= (1, 0.5), gamma = 0.2)


    


