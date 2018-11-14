
# coding: utf-8

# naive LSTM model trained with smooth data

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class DecoderRNN(nn.Module):
    def __init__(self, input_size, augmented_size, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.input_size = input_size
        self.augmented_size = augmented_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.cnn1 = nn.Conv1d(self.input_size, self.augmented_size//2 , kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.cnn2 = nn.Conv1d(self.augmented_size//2, self.augmented_size, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.lstm_1 = nn.LSTM(self.augmented_size, self.hidden_size//2, num_layers=1, bidirectional=True)
        self.lstm_2 = nn.LSTM(self.hidden_size, self.hidden_size//2, num_layers=1, bidirectional=True)
        self.lstm_3 = nn.LSTM(self.hidden_size, self.hidden_size//2, num_layers=1, bidirectional=True)
        self.lstm_4 = nn.LSTM(self.hidden_size, self.hidden_size//2, num_layers=1, bidirectional=True)
        self.lstm_5 = nn.LSTM(self.hidden_size, self.hidden_size//2, num_layers=1, bidirectional=True)
        self.lstm_6 = nn.LSTM(self.hidden_size, self.hidden_size//2, num_layers=1, bidirectional=True)
        self.lstm_7 = nn.LSTM(self.hidden_size, self.hidden_size//2, num_layers=1, bidirectional=True)
        # map the output of LSTM to the output space
        self.out = nn.Linear(self.hidden_size, self.output_size)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        self.hidden1 = self.init_hidden()
        self.hidden2 = self.init_hidden()
        self.hidden3 = self.init_hidden()
        self.hidden4 = self.init_hidden()
        self.hidden5 = self.init_hidden()
        self.hidden6 = self.init_hidden()
        self.hidden7 = self.init_hidden()

        output = self.cnn1(input.view(-1,3,1))
        output = self.relu1(output)
        output = self.cnn2(output)
        output = self.relu2(output)

        output, self.hidden1 = self.lstm_1(output.view(-1,1,self.augmented_size), self.hidden1)
        output_1 = output

        output, self.hidden2 = self.lstm_2(output, self.hidden2)
        output_2 = output

        output, self.hidden3 = self.lstm_3(output + output_1, self.hidden3)  # skip_connection 1
        output_3 = output

        output, self.hidden4 = self.lstm_4(output + output_2, self.hidden4)  # skip_connection 2
        output_4 = output

        output, self.hidden5 = self.lstm_5(output + output_3, self.hidden5)  # skip_connection 3
        output_5 = output

        output, self.hidden6 = self.lstm_6(output + output_4, self.hidden6)  # skip_connection 4

        output, self.hidden7 = self.lstm_7(output + output_5, self.hidden7)  # skip_connection 5
        
        output = self.out(output).view(-1,1)
        # output = self.softmax(output)
        return output

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_size // 2, device=device),
                torch.randn(2, 1, self.hidden_size // 2, device=device))
print(torch.cuda.is_available())
""" 
NOTE: 
Decoder RNN input of size (sentence_len, batch_size, input_dim) (scalar value)
Decoder RNN output of size (sentence_len, output_size)  batch_size = 1
"""

# In[5]:

import pickle

# load data from file

with open("/home/yiqin/2018summer_project/data/smooth_3d1s_augmented_small.pkl", "rb") as f:
    data= pickle.load(f)

train_X = []
train_Y = []
for i in range(len(data)):
    train_X.append(data[i][0])
    train_Y.append(data[i][1])
print(len(train_X))
print(len(train_Y))

with open("/home/yiqin/2018summer_project/data/smooth_3d_1start_val.pkl", "rb") as f:
    dic = pickle.load(f)
    val_X = dic["X"]
    val_Y = dic["Y"]
        
target_Tensor = train_Y
maximum_target = len(train_Y)

def penalty_loss(penalty, criterion, output, target):
    loss = 0
    for i in range(target.shape[0]):
        if int(target[i]) == 1:
            loss += penalty[0] * criterion(output[i], target[i])
        else:
            loss += penalty[1] * criterion(output[i], target[i])
    return loss


def train(input_tensor, target_tensor, decoder, decoder_optimizer, criterion, verbose = False, penalty = (1, 0.5)):
    decoder_optimizer.zero_grad()
    
    loss = 0
    
    
    decoder_output= decoder(input_tensor)
    
    if verbose:
        print("prediction score:", decoder_output.squeeze().detach().cpu().numpy())
        print("ground truch:", target_tensor.squeeze().cpu().numpy())

    loss += penalty_loss(penalty, criterion, decoder_output.squeeze(0), target_tensor.float())

    loss.backward()
    
    decoder_optimizer.step()
    

    return loss.item()

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

def trainIters(decoder, n_iters, print_every = 1000, plot_every = 100, learning_rate = 0.01, total_batch = maximum_target, gamma = 0.1):    
    start = time.time()
    
    plot_losses = []
    val_acc = []
    print_loss_total = 0
    plot_loss_total = 0
    
    best_acc = 0
    
    decoder_optimizer = optim.SGD(decoder.parameters(), lr = learning_rate)
    
    criterion = nn.SmoothL1Loss()
    
    scheduler = optim.lr_scheduler.StepLR(decoder_optimizer, step_size = total_batch, gamma = gamma)
    
    
    for iter in range(1, n_iters + 1):
        num = iter % total_batch
        verbose = (iter % print_every == 0)
        input_tensor = train_X[num].to(device).float()
        target_tensor = target_Tensor[num].to(device).float()
        #input_tensor = Variable(input_tensor, requires_grad = True)
        #print(input_tensor.shape, target_tensor.shape)
        if input_tensor.shape[0]<2:
            continue
        if input_tensor.shape[0] != target_tensor.shape[0]:
            continue
        
        loss = train(input_tensor, target_tensor, decoder, 
                     decoder_optimizer, criterion, verbose = verbose)
        print_loss_total += loss
        plot_loss_total += loss
        
        if iter % plot_every == 0:
            plot_losses.append(plot_loss_total / plot_every)
            plot_loss_total = 0

        if iter % print_every == 0:
            score = 0
            acc, one_acc, score = validate(decoder, val_X, val_Y)
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            
            val_acc.append((acc, one_acc))
            
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * print_every, print_loss_avg))
            
            print("validation accuracy:", acc)
            print("validation prediction accuracy:", one_acc)

            if(score > best_acc):
                torch.save(decoder.state_dict(), '/home/yiqin/2018summer_project/saved_model/Bi-LSTM-CNN_best.pt')
                best_acc = score
            print("best_score:", best_acc)
        
        if(iter%total_batch == 0 and iter > 0):    
            torch.save(decoder.state_dict(), '/home/yiqin/2018summer_project/saved_model/Bi-LSTM-CNN{}.pt'.format(iter//total_batch))
            
        
        scheduler.step()
    return plot_losses, val_acc

def validate(decoder, val_x, val_y, val_threshold = 0.5):

    count = 0
    total = 0
    total_1 = 0
    for i in range(len(val_x)):
        X = val_x[i].to(device).float()
        result = decoder(X).squeeze().cpu().detach().numpy()
        Y = val_y[i].squeeze()
        gt = (Y == 1).astype(int)
        pr = (result > 0.5).astype(int)
        
        count += np.sum(gt * pr)
        total += np.sum(gt)
        total_1 += np.sum(pr)
    score = (count / total) + (count / total_1)
    acc = str('%.4f'%((count / total * 100)) + "%")
    if total_1 == 0.0:
        one_acc = str('%.4f'%(0) + "%")
    else:
        one_acc = str('%.4f'%((count / total_1 * 100)) + "%")
    return acc, one_acc, score


input_size = 3
augmented_size = 32
hidden_size = 256
output_size = 1

decoder = DecoderRNN(input_size, augmented_size, hidden_size, output_size).to(device)

losses, val_acc  = trainIters(decoder, 150000, print_every=1000, learning_rate=1e-2, gamma=0.1)

dic = {}
dic["loss"] = losses
dic["acc"] = val_acc

f = open("/home/yiqin/2018summer_project/saved_model/Bi-LSTM-CNN_losses.pkl", "wb")
pickle.dump(dic, f)

