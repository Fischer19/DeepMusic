
# coding: utf-8

# In[1]:

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import pickle
device = torch.device(0 if torch.cuda.is_available() else "cpu")
from copy import deepcopy
import random


# In[2]:

class CrossValidator:
    def __init__(self, model, data, compute_acc, loss_function,
                 partition=5, decoder=None, batch_size=100, epochs=300, lr=1e-3):
        self.model=model
        self.data=data
        self.data_size=len(data)
        self.partition=partition
        self.decoder=decoder
        self.compute_acc=compute_acc
        self.train_data=[]
        self.val_X=[]
        self.val_Y=[]
        self.precision=0
        self.recall=0
        self.loss_history=[]
        self.loss_function=loss_function
        self.batch_size=batch_size
        self.epochs=epochs
        self.lr=lr
        
    def create_data(self, part):
        train_data=[]
        val_X=[]
        val_Y=[]
        cut=int(self.data_size/self.partition)
        for i, x in enumerate(self.data):
            if i<cut*part or i>=cut*(part+1):
                train_data.append([x[0],x[1]])
            else:
                val_X.append(x[0])
                val_Y.append(x[1])
        return train_data, val_X, val_Y
    
    def tensorize(self, p):
        p=np.array(p)
        p=torch.from_numpy(p).float()
        p=p.to(device)
        return p
    
    def create_batch(self, index):
        output_X=[]
        output_Y=[]
        for i in range(self.batch_size):
            if self.batch_size*index+i<len(self.train_data):
                output_X.append(self.train_data[self.batch_size*index+i][0])
                output_Y.append(self.train_data[self.batch_size*index+i][1])
        return output_X, output_Y
                
    
    def compute(self):
        for i in range(self.partition):
            self.train_data, self.val_X, self.val_Y = self.create_data(i)
            self.val_X=self.tensorize(self.val_X)
            self.val_Y=self.tensorize(self.val_Y)
            self.val_Y=self.val_Y.long()
            cur_model=deepcopy(self.model).to(device)
            optimizer=optim.Adam(cur_model.parameters(), lr=self.lr, weight_decay=5e-8)
            loss=0
            train_len=len(self.train_data)
            self.loss_history.append([])
            for j in range(self.epochs):
                random.shuffle(self.train_data)
                for k in range(int(train_len/self.batch_size)):
                    batch_X, batch_Y = self.create_batch(k)
                    batch_X = self.tensorize(batch_X)
                    batch_Y = self.tensorize(batch_Y)
                    batch_Y = batch_Y.long()
                    optimizer.zero_grad()
                    output_Y = cur_model(batch_X)
                    loss=self.loss_function(output_Y, batch_Y)
                    loss.backward()
                    optimizer.step()
                self.loss_history[i].append(loss.cpu().data)
                print("compeleted: ", float(i*self.epochs+j)/float(self.partition*self.epochs))
            output_Y = cur_model(self.val_X)
            if self.decoder!=None: output_Y=self.decoder(output_Y)
            temp_precision, temp_recall = self.compute_acc(output_Y, self.val_Y)
            self.precision+=temp_precision
            self.recall+=temp_recall
            print(self.precision, self.recall)
        self.precision /= float(self.partition)
        self.recall /= float(self.partition)
        return self.precision, self.recall, self.loss_history


# In[3]:

class CnnMusic(nn.Module):
    def __init__(self):
        super(CnnMusic, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 3, padding=2)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=2)
        self.conv3 = nn.Conv1d(128, 128, 3, padding=2)
        self.conv4 = nn.Conv1d(128, 64, 3, padding=2)
        self.conv5 = nn.Conv1d(64, 32, 5, padding=4)
        self.conv6 = nn.Conv1d(32, 2, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = x[:, :, :-2]
        x = self.conv2(x)
        x = x[:, :, :-2]
        x = self.conv3(x)
        x = x[:, :, :-2]
        x = self.conv4(x)
        x = x[:, :, :-2]
        x = self.conv5(x)
        x = x[:, :, :-4]
        x = self.conv6(x)
        return x


# In[4]:

model=CnnMusic()

def decode(p):
    p=torch.argmax(p, dim=1)
    return p

def compute_acc(prediction, ground_truth):
    #print("into func")
    prediction=prediction.cpu()
    ground_truth=ground_truth.cpu()
    b, l = prediction.shape
    cor=0
    tot1=0
    tot2=0
    for i in range(b):
        for j in range(l):
            if ground_truth[i][j]==1: tot1+=1
            if prediction[i][j]==1: tot2+=1
            if prediction[i][j]==1 and ground_truth[i][j]==1: cor+=1
        if i%100==0: print(i)
    #print("out func")
    return cor/max(1,tot2), cor/max(1,tot1)


# In[5]:

with open("/home/yiqin/2018summer_project/data/01_data_long.pkl", "rb") as f:
    dic = pickle.load(f)
    data_X = dic["X"]
    data_Y = dic["Y"]

data=[]
for i,x in enumerate(data_X):
    temp_x=np.zeros((3,880))
    temp_y=np.zeros((880))
    l=len(data_X[i])
    for j in range(880):
        temp_x[0][j]=data_X[i][j%l][1]-data_X[i][j%l][0]
        temp_x[1][j]=data_X[i][j%l][2]
        if (j%l)!=l-1:
            temp_x[2][j]=data_X[i][(j+1)%l][0]-data_X[i][j%l][1]
        else:
            temp_x[2][j]=random.random()
        temp_y[j]=data_Y[i][j%l]
    data.append([temp_x, temp_y])


# In[6]:

validator = CrossValidator(model, data, compute_acc=compute_acc, loss_function=nn.CrossEntropyLoss(), decoder=decode)
precision, recall, loss_history = validator.compute()


# In[7]:

import pickle
f=open('cnn_val.pk', 'wb')
pickle.dump(precision, f)
pickle.dump(recall, f)
pickle.dump(loss_history, f)
f.close()


# In[ ]:



