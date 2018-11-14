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
import copy
from torch.autograd import Variable
import argparse 

parser = argparse.ArgumentParser(description="BiLSTM-CNN training with cross validation")
parser.add_argument("-b", "--binary", default=False, type=bool, help="enable binary label for training")
parser.add_argument("-a", "--augment", default=0, help="number of augmentation multiple")
parser.add_argument("-t", "--train", default=False, type=bool, help="Train the model or cross validation")
parser.add_argument("-p", "--penalty", default=True, type=bool, help="Train without panelty loss")

device = torch.device(3 if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 40
CLIP = 10

def main():
    args = parser.parse_args()
    binary = args.binary
    train = args.train
    penalty_flag = args.penalty
    augment_data = int(args.augment)

    input_size = 3
    augmented_size = 32
    hidden_size = 256
    if binary:
        output_size = 2
    else:
        output_size = 1
    batch_size = 30
    batch_length = 100
    model = DecoderRNN(input_size, augmented_size, hidden_size, output_size, dropout_p = 0).to(device)
    cv = CrossValidator(model, partition=5, epochs=5, batch_size = batch_size, batch_length=batch_length, 
                        augment_data=augment_data, lr = 1e-2, print_every = 100, binary=binary, 
                        penalty=[1,0.4], train=train, penalty_flag=penalty_flag)
    losses, precision, recall = cv.compute()


    dic = {}
    dic["loss"] = losses
    dic["precision"] = recall
    dic["recall"] = precision

    f = open("/home/yiqin/2018summer_project/saved_model/Bi-LSTM-CNN_losses(cv_penalty_decay).pkl", "wb")
    pickle.dump(dic, f)
 
    
class DecoderRNN(nn.Module):
    def __init__(self, input_size, augmented_size, hidden_size, output_size, dropout_p = 0):
        super(DecoderRNN, self).__init__()
        self.input_size = input_size
        self.augmented_size = augmented_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.verbose = (self.dropout_p != 0)

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
        self.batch_size = input.shape[0]
        
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
        
        output = self.out(output).view(self.batch_size, -1,self.output_size)
        # output = self.softmax(output)
        return output

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_size // 2, device=device),
                torch.randn(2, 1, self.hidden_size // 2, device=device))
print(torch.cuda.is_available())

def validate(decoder, val_x, val_y, val_threshold = 0.5, binary= False):
    print(len(val_x))
    count = 0
    total = 0
    total_1 = 0
    
    #val_set = data_utils.TensorDataset(val_x, val_y)
    #val_loader=data_utils.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, drop_last=True, shuffle=True) 

    if binary:
        criterion = nn.CrossEntropyLoss()
        loss = 0
        for i in range(len(val_x)):
            X = val_x[i].to(device).float()
            result = decoder(X).squeeze().cpu().detach().numpy()
            pr = np.argmax(result, axis = 1)
            Y = val_y[i].squeeze().numpy()
            gt = (Y == 1).astype(int)
            count += np.sum(gt * pr)
            total += np.sum(gt)
            total_1 += np.sum(pr)
        recall = (count / total)
        precision = (count / total_1)
        F1 = 2 * (recall * precision)/(recall + precision)
        acc = str('%.4f'%((count / total * 100)) + "%")
        if total_1 == 0.0:
            one_acc = str('%.4f'%(0) + "%")
        else:
            one_acc = str('%.4f'%((count / total_1 * 100)) + "%")
    else:
        criterion = nn.SmoothL1Loss()
        loss = 0
        for i in range(len(val_x)):
            X = val_x[i].to(device).float()
            Y = val_y[i].to(device)
            decoder_output= decoder(X)
            loss += penalty_loss([1,1], criterion, decoder_output.squeeze(0), Y.float())
            result = decoder_output.squeeze().cpu().detach().numpy()
            pr = [int(result[0] > 0.5)]
            for j in range(1, len(result) - 1):
                if result[j] > 0.5 and result[j] > result[j-1] and result[j] > result[j+1]:
                    pr.append(1)
                else:
                    pr.append(0)
            pr.append(0)
            pr = np.array(pr).astype(int)
            Y = Y.cpu().squeeze().numpy()
            gt = (Y == 1).astype(int)
            count += np.sum(gt * pr)
            total += np.sum(gt)
            total_1 += np.sum(pr)
        recall = (count / total) 
        precision = (count / total_1)
        F1 = 2 * (recall * precision)/(recall + precision)
        acc = str('%.4f'%((count / total * 100)) + "%")
        if total_1 == 0.0:
            one_acc = str('%.4f'%(0) + "%")
        else:
            one_acc = str('%.4f'%((count / total_1 * 100)) + "%")
    return acc, one_acc, F1, loss/len(val_x)
from copy import deepcopy

def penalty_loss(penalty, criterion, output, target, binary=False):
    loss = 0
    batch_size = target.shape[0]
    for j in range(target.shape[0]):
        for i in range(target.shape[1]):
            if int(target[j, i]) == 1:
                if binary:
                    loss += penalty[0] * criterion(output[j, i].unsqueeze(dim = 0), target[j, i].long().unsqueeze(dim = 0))
                else:
                    loss += penalty[0] * criterion(output[j, i], target[j, i])
            else:
                if binary:
                    loss += penalty[1] * criterion(output[j, i].unsqueeze(dim = 0), target[j, i].long().unsqueeze(dim = 0))
                else:
                    loss += penalty[1] * criterion(output[j, i], target[j, i])
    return loss/batch_size

def train(input_tensor, target_tensor, decoder, decoder_optimizer, criterion, penalty = [1, 0.5], binary = False):
    decoder_optimizer.zero_grad()
    
    loss = 0
    
    
    decoder_output= decoder(input_tensor)
    
    #if verbose:
    #    print("prediction score:", decoder_output.squeeze().detach().cpu().numpy())
    #    print("ground truch:", target_tensor.squeeze().cpu().numpy())
    loss += penalty_loss(penalty, criterion, decoder_output.squeeze(0), target_tensor.float(), binary)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), CLIP)
    decoder_optimizer.step()
    

    return loss.item()

def factorize(data_X, data_Y, size, batch_size, batch_length, sentence = 5):
    new_X = []
    new_Y = []
    for i in range(len(data_X)):
        X, Y = data_X[i], data_Y[i]
        flag = []
        for loc, j in enumerate(Y):
            if j == 1:
                flag.append(loc)
        prev = 0
        for j in range(sentence -1, len(flag), sentence):
            new_X.append(pad(torch.from_numpy(X[prev:flag[j]]), batch_length))
            new_Y.append(pad(torch.from_numpy(Y[prev:flag[j]]), batch_length))
            prev = flag[j]
    batch_X = []
    batch_Y = []
    """for i in range(0, len(new_X), batch_size):
        if (i + batch_size) > (len(new_X) - 1):
            break
        batch_x = []
        batch_y = []
        for j in range(batch_size):
            batch_x.append(pad(new_X[i + j], batch_length))
            batch_y.append(pad(new_Y[i + j], batch_length))
        batch_X.append(torch.stack(batch_x))
        batch_Y.append(torch.stack(batch_y))
    return batch_X, batch_Y"""
    print(torch.stack(new_X).shape)
    print(torch.stack(new_Y).shape)
    return torch.stack(new_X), torch.stack(new_Y)

def pad(vector, pad, dim=0):
    pad_size=list(vector.shape)
    #print(pad_size)
    pad_size[dim]=pad-vector.size(dim)
    #print(pad_size[dim])
    if pad_size[dim]<0:
        print("FATAL ERROR: pad_size=100 not enough!")
        print(pad, pad_size[dim])
    return torch.cat([vector, torch.zeros(*pad_size).type(vector.type())], dim=dim)


import torch.utils.data as data_utils
class CrossValidator:
    def __init__(self, model, partition=5, decoder=None, batch_size=BATCH_SIZE, batch_length = 100, epochs=10, lr=1e-2, 
                 augment_data=0, print_every = 1000, plot_every = 100, gamma = 0.1, penalty = [1, 0.5], 
                 binary= False, train=False, penalty_flag=True):
        self.model=model
        self.data_X = []
        self.data_Y = []
        self.augment_data_size=augment_data
        with open("/home/yiqin/2018summer_project/data/smooth_3d1s_augmented.pkl", "rb") as f:
            data= pickle.load(f)
            for i in range(len(data)):
                self.data_X.append(data[i][0])
                self.data_Y.append(data[i][1])
            
        
        self.data_size = len(self.data_X)
        self.partition=partition
        self.decoder=decoder
        self.train_X=[]
        self.train_Y=[]
        self.val_X=[]
        self.val_Y=[]
        self.precision_history=[]
        self.recall_history=[]
        self.loss_history=[]
        self.val_loss_history=[]
        self.best_acc = 0
        self.batch_size=batch_size
        self.batch_length=batch_length
        self.epochs=epochs
        self.lr=lr
        self.gamma = gamma
        self.penalty = penalty
        self.print_every = print_every
        self.plot_every = plot_every
        self.binary = binary
        self.train = train
        self.penalty_flag = penalty_flag
        
    def create_data(self, part, binary = False):
        train_X=[]
        train_Y=[]
        val_X=[]
        val_Y=[]
        cut=int(self.data_size/self.partition)
        for i in range(self.data_size):
            if i<cut*part or i>=cut*(part+1):
                train_X.append(np.array(self.data_X[i]))
                if binary:
                    train_Y.append((np.array(self.data_Y[i]) == 1).astype(int).squeeze())
                else:
                    train_Y.append(np.array(self.data_Y[i]))
            else:
                val_X.append(np.array(self.data_X[i]))
                if binary:
                    val_Y.append((np.array(self.data_Y[i]) == 1).astype(int).squeeze())
                else:
                    val_Y.append(np.array(self.data_Y[i]))
                
        new_X = []
        for loc, item in enumerate(train_X):
            x = copy.deepcopy(item)
            x[:,0] = x[:,0] + 0.5
            new_X.append(x)
        train_X += new_X
        train_Y += train_Y
        augmented_X = []
        augmented_Y = []
        augmented_X += train_X
        augmented_Y += train_Y
        for i in range(self.augment_data_size):
            new_X = []
            print(i)
            if i == (self.augment_data_size //2):
                continue
            for loc, item in enumerate(train_X):
                x = copy.deepcopy(item)
                x[:,2] = x[:,2] + i - (self.augment_data_size // 2)
                new_X.append(x)
            augmented_X += new_X
            augmented_Y += train_Y

        train_Y = augmented_Y
        train_X = augmented_X
        return train_X, train_Y, val_X, val_Y
    
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
    
    def penalty_decay(self, penalty, epoch, decay):
        penalty[1] = (1+decay)**epoch * penalty[1]
        return penalty

    def compute(self):
        for i in range(self.partition):
            if self.train and i > 0:
                break
            start = time.time()
            temptrain_X, temptrain_Y, tempval_X, tempval_Y = self.create_data(i, self.binary)
            print(len(temptrain_X), len(temptrain_Y), len(tempval_X), len(tempval_Y))
            self.train_X, self.train_Y = factorize(temptrain_X, temptrain_Y, self.augment_data_size, self.batch_size, self.batch_length, 5)
            self.val_X, self.val_Y = factorize(tempval_X, tempval_Y, self.augment_data_size, self.batch_size, self.batch_length, 5)
            train_set = data_utils.TensorDataset(self.train_X, self.train_Y)
            train_loader=data_utils.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, drop_last=True, shuffle=True) 
            print(len(train_loader))
            
            print(i, "phase 1 completed.")
            
            cur_model = deepcopy(self.model).to(device)
            
            optimizer = optim.SGD(cur_model.parameters(), lr = self.lr, weight_decay = 1e-5)
    
            if self.binary:
                criterion = nn.CrossEntropyLoss()
            else:
                criterion = nn.SmoothL1Loss()
        
            penalty = deepcopy(self.penalty)
            self.best_acc = 0

            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = self.gamma)

            print_loss_total = 0
            plot_loss_total = 0
            
            
            for j in range(self.epochs):
                if self.penalty_flag:
                    penalty = self.penalty_decay(penalty, j, decay = 0.05)
                else:
                    penalty = [1,1]
                for num, (train_X, train_Y)in enumerate(train_loader):
                    input_tensor = train_X.to(device).float()
                    target_tensor = train_Y.to(device).float()
                    loss = train(input_tensor, target_tensor, cur_model, decoder_optimizer=optimizer, criterion= criterion, penalty = penalty, binary=self.binary)
                    print(loss)
                    print_loss_total += loss
                    plot_loss_total += loss
                    
                    if num%self.plot_every == 0:
                        plot_loss_avg = plot_loss_total / self.plot_every
                        plot_loss_total = 0
                        self.loss_history.append(plot_loss_avg)

                    if (num) % self.print_every == 0:
                        acc, one_acc, score, val_loss = validate(cur_model, self.val_X, self.val_Y, binary= self.binary)
                        self.precision_history.append(acc[:-1])
                        self.recall_history.append(one_acc[:-1])
                        self.val_loss_history.append(val_loss)
                        print_loss_avg = print_loss_total / self.print_every
                        print_loss_total = 0
                        print("partition %i epoch %i"%(i,j))
                        p = self.timeSince(start, (1+num+j*len(train_loader)) / (self.epochs * len(train_loader)))
                        print('%s (%d %d%%) %.4f' % (p, num + 1, (num + 1 + j*len(train_loader)) / (self.epochs * len(train_loader)) * self.print_every,
                                                     print_loss_avg))
                        print("recall:", acc)
                        print("precision:", one_acc)
                        print("validation loss:", val_loss)
                        if(score > self.best_acc):
                        #    torch.save(cur_model.state_dict(), '/home/yiqin/2018summer_project/saved_model/Bi-LSTM-CNN_best(cv).pt')
                            self.best_acc = score
                        print("best_score:", self.best_acc)

                scheduler.step()
                if self.train:
                    torch.save(cur_model.state_dict(), '/home/yiqin/2018summer_project/saved_model/Bi-LSTM-CNN(train).pt')
                dic = {}
                dic["recall"] = self.precision_history
                dic["precision"] = self.recall_history
                dic["loss"] = self.loss_history
                dic["val"] = self.val_loss_history
                f = open("/home/yiqin/2018summer_project/saved_model/Bi-LSTM-CNN_losses(val_loss).pkl", "wb")
                pickle.dump(dic, f)
                f.close()


        return self.loss_history, self.precision_history, self.recall_history


if __name__ == "__main__":
    main()