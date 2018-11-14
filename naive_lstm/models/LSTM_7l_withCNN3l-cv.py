
# coding: utf-8

# naive LSTM model trained with smooth data

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pickle
import math
import time
import numpy as np
from torch.autograd import Variable

device = torch.device(2 if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
CLIP = 10

class DecoderRNN(nn.Module):
    def __init__(self, input_size, augmented_size, hidden_size, output_size, dropout_p):
        super(DecoderRNN, self).__init__()
        self.input_size = input_size
        self.augmented_size = augmented_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.verbose = (self.dropout_p != 0)

        self.cnn1 = nn.Conv1d(self.input_size, self.augmented_size//4 , kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.cnn2 = nn.Conv1d(self.augmented_size//4, self.augmented_size//2, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.cnn3 = nn.Conv1d(self.augmented_size//2, self.augmented_size, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.lstm_1 = nn.LSTM(self.augmented_size, self.hidden_size//2, num_layers=1, bidirectional=True)
        self.lstm_2 = nn.LSTM(self.hidden_size, self.hidden_size//2, num_layers=1, bidirectional=True)
        self.lstm_3 = nn.LSTM(self.hidden_size, self.hidden_size//2, num_layers=1, bidirectional=True)
        self.lstm_4 = nn.LSTM(self.hidden_size, self.hidden_size//2, num_layers=1, bidirectional=True)
        self.lstm_5 = nn.LSTM(self.hidden_size, self.hidden_size//2, num_layers=1, bidirectional=True)
        self.lstm_6 = nn.LSTM(self.hidden_size, self.hidden_size//2, num_layers=1, bidirectional=True)
        self.lstm_7 = nn.LSTM(self.hidden_size, self.hidden_size//2, num_layers=1, bidirectional=True)
        self.dropout1 = nn.Dropout(self.dropout_p)
        self.dropout2 = nn.Dropout(self.dropout_p)
        self.dropout3 = nn.Dropout(self.dropout_p)
        self.dropout4 = nn.Dropout(self.dropout_p)
        self.dropout5 = nn.Dropout(self.dropout_p)
        self.dropout6 = nn.Dropout(self.dropout_p)
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
        output = self.cnn3(output)
        output = self.relu3(output)
        
        output, self.hidden1 = self.lstm_1(output.view(-1,1,self.augmented_size), self.hidden1)
        if self.verbose:
            output = self.dropout1(output)
        output_1 = output

        output, self.hidden2 = self.lstm_2(output, self.hidden2)
        if self.verbose:
            output = self.dropout2(output)
        output_2 = output

        output, self.hidden3 = self.lstm_3(output + output_1, self.hidden3)  # skip_connection 1
        if self.verbose:
            output = self.dropout3(output)
        output_3 = output

        output, self.hidden4 = self.lstm_4(output + output_2, self.hidden4)  # skip_connection 2
        if self.verbose:
            output = self.dropout4(output)
        output_4 = output

        output, self.hidden5 = self.lstm_5(output + output_3, self.hidden5)  # skip_connection 3
        if self.verbose:
            output = self.dropout5(output)
        output_5 = output

        output, self.hidden6 = self.lstm_6(output + output_4, self.hidden6)  # skip_connection 4
        if self.verbose:
            output = self.dropout6(output)
        output, self.hidden7 = self.lstm_7(output + output_5, self.hidden7)  # skip_connection 5
        
        output = self.out(output).view(-1,1)
        # output = self.softmax(output)
        return output

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_size // 2, device=device),
                torch.randn(2, 1, self.hidden_size // 2, device=device))
print(torch.cuda.is_available())


def penalty_loss(penalty, criterion, output, target):
    loss = 0
    for i in range(target.shape[0]):
        if int(target[i]) == 1:
            loss += penalty[0] * criterion(output[i], target[i])
        else:
            loss += penalty[1] * criterion(output[i], target[i])
    return loss

def validate(decoder, val_x, val_y, val_threshold = 0.5):

    count = 0
    total = 0
    total_1 = 0
    
    #val_set = data_utils.TensorDataset(val_x, val_y)
    #val_loader=data_utils.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, drop_last=True, shuffle=True) 

    
    for i in range(len(val_x)):
        X = val_x[i].to(device).float()
        result = decoder(X).squeeze().cpu().detach().numpy()
        pr = [int(result[0] > 0.5)]
        for j in range(1, len(result) - 1):
            if result[j] > 0.5 and result[j] > result[j-1] and result[j] > result[j+1]:
                pr.append(1)
            else:
                pr.append(0)
        pr.append(0)
        pr = np.array(pr).astype(int)
        Y = val_y[i].squeeze().numpy()
        gt = (Y == 1).astype(int)
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
from copy import deepcopy

def penalty_loss(penalty, criterion, output, target):
    loss = 0
    for i in range(target.shape[0]):
        if int(target[i]) == 1:
            loss += penalty[0] * criterion(output[i], target[i])
        else:
            loss += penalty[1] * criterion(output[i], target[i])
    return loss

def train(input_tensor, target_tensor, decoder, decoder_optimizer, criterion, penalty = (1, 0.5)):
    decoder_optimizer.zero_grad()
    
    loss = 0
    
    
    decoder_output= decoder(input_tensor)
    
    #if verbose:
    #    print("prediction score:", decoder_output.squeeze().detach().cpu().numpy())
    #    print("ground truch:", target_tensor.squeeze().cpu().numpy())

    loss += penalty_loss(penalty, criterion, decoder_output.squeeze(0), target_tensor.float())
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), CLIP)
    decoder_optimizer.step()
    

    return loss.item()

def factorize(data_X, data_Y, flag):
    if flag:
        new_X = []
        new_Y = []
        for i in range(len(data_X)):
            X, Y = data_X[i], data_Y[i]
            flag = []
            for loc, j in enumerate(Y):
                if j == 1:
                    flag.append(loc)
            prev = 0
            for j in range(4, len(flag), 5):
                new_X.append(torch.from_numpy(X[prev:flag[j]]))
                new_Y.append(torch.from_numpy(Y[prev:flag[j]]))
                prev = flag[j]
    return new_X, new_Y

class CrossValidator:
    def __init__(self, model, partition=5, decoder=None, batch_size=BATCH_SIZE, epochs=10, lr=1e-2, 
                 augment_data=True, print_every = 1000, plot_every = 100, gamma = 0.1, penalty = [1,0.5]):
        self.model=model
        self.data_X = []
        self.data_Y = []
        with open("/home/yiqin/2018summer_project/data/smooth_3d1s_augmented.pkl", "rb") as f:
            data= pickle.load(f)
            for i in range(len(data)):
                self.data_X.append(data[i][0])
                self.data_Y.append(data[i][1])
        self.data_size = len(data)
        self.partition=partition
        self.decoder=decoder
        self.train_X=[]
        self.train_Y=[]
        self.val_X=[]
        self.val_Y=[]
        self.precision_history=[]
        self.recall_history=[]
        self.loss_history=[]
        self.best_acc = 0
        self.batch_size=batch_size
        self.epochs=epochs
        self.lr=lr
        self.augment_data_flag=augment_data
        self.gamma = gamma
        self.penalty = penalty
        self.print_every = print_every
        self.plot_every = plot_every
        
        
    def create_data(self, part):
        train_X=[]
        train_Y=[]
        val_X=[]
        val_Y=[]
        cut=int(self.data_size/self.partition)
        for i in range(self.data_size):
            if i<cut*part or i>=cut*(part+1):
                train_X.append(np.array(self.data_X[i]))
                train_Y.append(np.array(self.data_Y[i]))
            else:
                val_X.append(np.array(self.data_X[i]))
                val_Y.append(np.array(self.data_Y[i]))
        return train_X, train_Y, val_X, val_Y
    
    def penalty_decay(self, penalty, epoch, decay):
        penalty[1] = (1-decay)**epoch * penalty[1]
        return penalty

    
    def tensorize(self, p):
        p=np.array(p)
        p=torch.from_numpy(p).float()
        p=p.to(device)
        return p
    
    def asMinutes(self, s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)


    def timeSince(self,since, percent):
        now = time.time()
        s = now - since
        es = s / (percent)
        rs = es - s
        return '%s (- %s)' % (self.asMinutes(s), self.asMinutes(rs))

    def compute(self):
        for i in range(self.partition):
            start = time.time()
            temptrain_X, temptrain_Y, tempval_X, tempval_Y = self.create_data(i)
            self.train_X, self.train_Y = factorize(temptrain_X, temptrain_Y, self.augment_data_flag)
            self.val_X, self.val_Y = factorize(tempval_X, tempval_Y, self.augment_data_flag)
            print(i, "phase 1 completed.")
            
            cur_model = deepcopy(self.model).to(device)
            
            optimizer = optim.SGD(cur_model.parameters(), lr = self.lr, weight_decay = 1e-5)
    
            criterion = nn.SmoothL1Loss()

            #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = self.gamma)

            print_loss_total = 0
            plot_loss_total = 0
            
            
            for j in range(self.epochs):
                for num in range(1, len(self.train_X) + 1):
                    input_tensor = self.train_X[num - 1].to(device).float()
                    target_tensor = self.train_Y[num - 1].to(device).float()
                    #self.penalty = self.penalty_decay(self.penalty, j, 0.05)
                    loss = train(input_tensor, target_tensor, cur_model, decoder_optimizer=optimizer, criterion= criterion, penalty = self.penalty)
                    print_loss_total += loss
                    plot_loss_total += loss
                    
                    if num%self.plot_every == 0:
                        plot_loss_avg = plot_loss_total / self.plot_every
                        plot_loss_total = 0
                        self.loss_history.append(plot_loss_avg)

                    if num % self.print_every == 0:
                        acc, one_acc, score = validate(cur_model, self.val_X, self.val_Y)
                        self.precision_history.append(acc[:-1])
                        self.recall_history.append(one_acc[:-1])
                        print_loss_avg = print_loss_total / self.print_every
                        print_loss_total = 0
                        print("epoch %i"%j)
                        p = self.timeSince(start, (num+j*len(self.train_X)) / (self.epochs * len(self.train_X)))
                        print('%s (%d %d%%) %.4f' % (p, num, num / (self.epochs * len(self.train_X)) * self.print_every,
                                                     print_loss_avg))
                        print("recall:", acc)
                        print("precision:", one_acc)
                        if(score > self.best_acc):
                            #torch.save(cur_model.state_dict(), '/home/yiqin/2018summer_project/saved_model/Bi-LSTM-CNN_best(cv).pt')
                            self.best_acc = score
                        print("best_score:", self.best_acc)

                #scheduler.step()
                #torch.save(cur_model.state_dict(), '/home/yiqin/2018summer_project/saved_model/Bi-LSTM-CNN3l(cv){}-{}.pt'.format(i,j))
                
        return self.loss_history, self.precision_history, self.recall_history

input_size = 3
augmented_size = 32
hidden_size = 128
output_size = 1
model = DecoderRNN(input_size, augmented_size, hidden_size, output_size, dropout_p = 0).to(device)
#model.load_state_dict(torch.load('/home/yiqin/2018summer_project/saved_model/Bi-LSTM-CNN4l(cv)0-0.pt'))
cv = CrossValidator(model, partition=5, epochs=1, penalty = [1, 0.5])
losses, precision, recall = cv.compute()


dic = {}
dic["loss"] = losses
dic["precision"] = recall
dic["recall"] = precision

f = open("/home/yiqin/2018summer_project/saved_model/Bi-LSTM-CNN3l_losses(cv).pkl", "wb")
pickle.dump(dic, f)

