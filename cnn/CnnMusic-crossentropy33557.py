
# coding: utf-8

# In[1]:

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from torch.autograd import Variable
import pickle
device = torch.device(0 if torch.cuda.is_available() else "cpu")


# In[2]:

class CnnMusic(nn.Module):
    def __init__(self):
        super(CnnMusic, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 3, padding=2)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=2)
        self.conv3 = nn.Conv1d(128, 128, 5, padding=4)
        self.conv4 = nn.Conv1d(128, 64, 5, padding=4)
        self.conv5 = nn.Conv1d(64, 32, 7, padding=6)
        self.conv6 = nn.Conv1d(32, 2, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = x[:, :, :-2]
        x = self.conv2(x)
        x = x[:, :, :-2]
        x = self.conv3(x)
        x = x[:, :, :-4]
        x = self.conv4(x)
        x = x[:, :, :-4]
        x = self.conv5(x)
        x = x[:, :, :-6]
        x = self.conv6(x)
        return x


# In[3]:

model=CnnMusic()
x=torch.from_numpy(np.random.randn(1,3,880)).float()
print(x.shape)
y=model(x)
print(y.shape)


# In[4]:

with open("/home/yiqin/2018summer_project/data/01_data_long.pkl", "rb") as f:
    dic = pickle.load(f)
    train_X = dic["X"]
    train_Y = dic["Y"]
print(len(train_X))
print(len(train_Y))


# In[5]:

data=[]
for i, x in enumerate(train_X):
    temp=[]
    temp.append(train_X[i])
    temp.append(train_Y[i])
    data.append(temp)
import random
#random.shuffle(data)
print(len(data))
train_data=data[:1300]
val_data=data[1300:]


# In[6]:

val_size=len(val_data)
val_x=np.zeros((val_size,3,880))
val_y=np.zeros((val_size,880))
val_len=[]
for i in range(val_size):
    val_len.append(len(val_data[i][0]))
    l=len(val_data[i][0])
    for j in range(880):
        val_x[i][0][j]=val_data[i][0][j%l][1]-val_data[i][0][j%l][0]
        val_x[i][1][j]=val_data[i][0][j%l][2]
        if (j%l)!=l-1:
            val_x[i][2][j]=val_data[i][0][(j+1)%l][0]-val_data[i][0][j%l][1]
        else:
            val_x[i][2][j]=random.random()
        val_y[i][j]=val_data[i][1][j%l]
#print(val_x)
#print(val_y)
val_x=Variable(torch.from_numpy(val_x).float().to(device))


# In[7]:

def make_batch(index, batch_size):
    s=index*batch_size
    xin=np.zeros((batch_size,3,880))
    yin=np.zeros((batch_size,880))
    for i in range(batch_size):
        l=len(train_data[s+i][0])
        for j in range(880):
            xin[i][0][j]=train_data[s+i][0][j%l][1]-train_data[s+i][0][j%l][0]
            xin[i][1][j]=train_data[s+i][0][j%l][2]
            if (j%l)!=l-1:
                xin[i][2][j]=train_data[s+i][0][(j+1)%l][0]-train_data[s+i][0][j%l][1]
            else:
                xin[i][2][j]=random.random()
            yin[i][j]=train_data[s+i][1][j%l]
    return xin, yin


# In[8]:

model=CnnMusic()
lr=1e-3
decay=5e-8
optimizer=optim.Adam(model.parameters(),
                     lr=lr,
                     weight_decay=decay)
batch_size=100
epochs=500
loss=0
data_len=len(train_data)
w=torch.Tensor([0.2,0.9]).to(device)
criterion = nn.CrossEntropyLoss(weight=w)


# In[9]:

def compute_acc(prediction, gt):
    p=torch.argmax(prediction, dim=1)
    #print(p)
    #print(gt)
    b, l = p.shape
    cor1=0
    tot1=0
    cor2=0
    tot2=0
    for i in range(b):
        for j in range(l):
            if gt[i][j]==1:
                tot1+=1
                if p[i][j]==1: cor1+=1
            if p[i][j]==1: 
                tot2+=1
                if gt[i][j]==1: cor2+=1
    return cor2/max(1,tot2), cor1/max(1,tot1)


# In[10]:

model=model.to(device)
best_model=None
best_val_acc=0
best_all_acc=0
for i in range(epochs):
    random.shuffle(train_data)
    for j in range(int(data_len/batch_size)):
        input_x, input_y = make_batch(j, batch_size)
        #input_x, input_y = make_batch(0,2) #overfit
        input_x=Variable(torch.from_numpy(input_x).float().to(device))
        input_y=Variable(torch.from_numpy(input_y).long().to(device))
        optimizer.zero_grad()
        output_y=model(input_x)
        loss=criterion(output_y, input_y)
        loss.backward()
        optimizer.step()
    if i%100==0: print(loss)

    output_val_y=model(val_x)
    all_acc, val_acc=compute_acc(output_val_y, val_y)
    if val_acc>best_val_acc and all_acc>best_all_acc and i>5:
        best_model = torch.save(model.state_dict(), "cnn_ce_best_model33557.pt")
        best_val_acc = val_acc
    print("current val acc:", all_acc, val_acc)
'''
#overfit
input_x, input_y = make_batch(0,2)
input_x=Variable(torch.from_numpy(input_x).float().to(device))
input_y=Variable(torch.from_numpy(input_y).long().to(device))
output_y=model(input_x)
print(input_y)
print(output_y)
print(compute_acc(output_y, input_y))
'''


# In[ ]:




# In[12]:

def generate_test():
    print(val_x.shape)
    test_y=model(val_x)
    
    p=torch.argmax(test_y, dim=1)
    print(p.shape)
    n, l=p.shape
    p=p.cpu().numpy()
    output=[]
    for i in range(n):
        output.append(p[i][:val_len[i]])
    return output
    
print(generate_test())


# In[13]:

output_y = generate_test()
output = []
output.append(np.array(output_y))
f = open("sample_prediction.pkl", "wb")
pickle.dump(output_y, f)
f.close()


# In[ ]:



