from center_loss import CenterLoss
from AutoCoV import AE
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from scipy.stats import entropy
from collections import Counter
import os
import numpy as np
import random
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

import sys

def count_to_prob(count_list):
    
    _count = Counter(count_list)
    
    if len(_count) == 1:
        prob = [1]
    else:
        prob = []
        _count_value = _count.values()
        for num in _count_value:
            prob.append(num/sum(_count_value))
        
    return prob

class COVID(Dataset):
    def __init__(self, data, labels):
        super().__init__()
        self.data = torch.from_numpy(data) # numpy array to torch tensor
        self.labels = torch.from_numpy(labels).type(torch.long) # numpy array to torch tensor

    def __getitem__(self, index):
        
        return self.data[index], self.labels[index]
    
    def __len__(self):
        return len(self.data)

def clade_num(clade_list):
    
    clade_num_list = []
    for clade in clade_list:
        if clade == 'S':
            clade_num_list.append(0)
        elif clade == 'L':
            clade_num_list.append(1)
        elif clade == 'V':
            clade_num_list.append(2)
        elif clade == 'G':
            clade_num_list.append(3)
        elif clade == 'GR':
            clade_num_list.append(4)
        else:
            clade_num_list.append(5)
    clade_num_input = np.array(clade_num_list)[:, np.newaxis]

    return np.array(clade_num_list)

def date_num(date_list):
    
    date_num_list = []
    for date in date_list:
        if date == 0:
            date_num_list.append(0)
        elif date == 1:
            date_num_list.append(1)
        else:
            date_num_list.append(2)
    date_num_input = np.array(date_num_list)[:, np.newaxis]

    return np.array(date_num_list)

def date_class(date):
   
    if date in ['12', '01', '02']:
        return 0
    elif date in ['03']:
        return 1
    else:
        return 2

def region_num(date_list):
    date_num_list = []
    for date in date_list:
        if date == 'North America':
            date_num_list.append(0)
        elif date == 'Asia':
            date_num_list.append(1)
        elif date == 'Oceania':
            date_num_list.append(2)
        elif date == 'Europe':
            date_num_list.append(3)
        elif date == 'Africa':
            date_num_list.append(4)
        else:
            date_num_list.append(5)
    date_num_input = np.array(date_num_list)[:, np.newaxis]
    return np.array(date_num_list)


#######################################################################
# Load Data & Preprocessing

train_df = pd.read_csv(sys.argv[1], sep = '\t', index_col = 0)
x_train = train_df.iloc[:,:-5]
y_train = train_df['Region'] # for spatial dynamics. In case of temporal dynamics, use 'Date_Class'

val_df = pd.read_csv(sys.argv[2], sep = '\t', index_col = 0)
x_val = val_df.iloc[:,:-5]
y_val = val_df['Region']

test_df = pd.read_csv(sys.argv[3], sep = '\t', index_col = 0)
x_test = test_df.iloc[:,:-5]
y_test = test_df['Region']


entropy_list = []
for col in x_train.columns[1:]:

    prob = count_to_prob(list(x_train[col]))
    entro = entropy(prob, base = 2)
    if entro >= 0.2:
        entropy_list.append(col)

x_train = x_train[[0] + entropy_list]
x_train_norm = x_train.iloc[:,1:].div(x_train.iloc[:,1:].sum(axis=1), axis=0)
scaler = StandardScaler()
scaler.fit(x_train_norm)
x_train_norm_scaler = scaler.transform(x_train_norm)

x_val = x_val[[0] + entropy_list]
x_val_norm = x_val.iloc[:,1:].div(x_val.iloc[:,1:].sum(axis=1), axis=0)
x_val_norm_scaler = scaler.transform(x_val_norm)

x_test = x_test[[0] + entropy_list]
x_test_norm = x_test.iloc[:,1:].div(x_test.iloc[:,1:].sum(axis=1), axis=0)
x_test_norm_scaler = scaler.transform(x_test_norm)

#######################################################################


####################################################################
### HYPER PARAMETERS ###

num_epochs = 200
BATCH_SIZE = 128
FEATURE_LEN = len(entropy_list)
NUM_CLASSES = len(set(region_num(y_train))) # for spatial dynamics. In case of temporal dynamics, use 'len(set(date_num(y_train)))'
DROP_OUT_RATE = 0.2
learning_rate = 0.01
lr_cent = 0.5
seed = 42
gpu = 0 # -1 for cpu // 0, 1, ../... : gpu number
#########################################################################

device = torch.device("cuda:"+str(gpu) if gpu != -1 else "cpu")
    
# Dataset
train_dataset = COVID(np.tanh(x_train_norm_scaler, dtype = np.float32)
                      , region_num(y_train)) ### INPUT
val_dataset = COVID(np.tanh(x_val_norm_scaler, dtype = np.float32)
                      , region_num(y_val)) ### INPUT
test_dataset = COVID(np.tanh(x_test_norm_scaler, dtype = np.float32)
                  , region_num(y_test)) ### INPUT

# Data Loder
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)



torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


model = AE().to(device)

reconstruction_loss = nn.MSELoss()
classification_loss = nn.CrossEntropyLoss()
center_loss = CenterLoss(num_classes = NUM_CLASSES, feat_dim=2, use_gpu=(True if gpu != -1 else False), gpu_num=gpu)

params = list(model.parameters()) + list(center_loss.parameters())

optimizer = torch.optim.Adam(params, lr = learning_rate)


def train(epoch):
    model.train()
    train_loss = 0.0

    true_label_list = []
    pred_label_list = []

    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        recon_batch, pred, hidden = model(data)
        pred_labels = np.argmax(pred.detach().cpu().numpy(), axis=1)

        loss = reconstruction_loss(recon_batch, data) + \
                classification_loss(pred, labels) + \
                center_loss(hidden, labels)    
        optimizer.zero_grad()
        loss.backward()
        
        # multiple (1./alpha) in order to remove the effect of alpha on updating centers
        for param in center_loss.parameters():
            param.grad.data *= (lr_cent / (alpha_cent * learning_rate))
        train_loss += loss.item()
        optimizer.step()

        true_label_list += list(labels.cpu().numpy())
        pred_label_list += list(pred_labels)


    print('====> Epoch: {} Average loss: {:.4f}\tAccuracy: {:.4f}\tF1: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset),
            accuracy_score(true_label_list, pred_label_list),
            f1_score(true_label_list, pred_label_list, average='weighted')
    ))

    return accuracy_score(true_label_list, pred_label_list), f1_score(true_label_list, pred_label_list, average='weighted')

def val(epoch):
    model.eval()
    val_loss = 0

    true_label_list = []
    pred_label_list = []

    with torch.no_grad():
        for i, (data, labels) in enumerate(val_loader):
            data = data.to(device)
            labels = labels.to(device)
            recon_batch, pred, hidden = model(data)
            pred_labels = np.argmax(pred.detach().cpu().numpy(), axis=1)
            loss = reconstruction_loss(recon_batch, data) + \
                classification_loss(pred, labels) + \
                center_loss(hidden, labels)  

            val_loss += loss.item()
            true_label_list += list(labels.cpu().numpy())
            pred_label_list += list(pred_labels)

    val_loss /= len(val_loader.dataset)
    print('====> Validation set loss: {:.4f}\tAccuracy: {:.4f}\tF1: {:.4f}'.format(val_loss,
                   accuracy_score(true_label_list, pred_label_list),
            f1_score(true_label_list, pred_label_list, average='weighted')
    ))

    return accuracy_score(true_label_list, pred_label_list), f1_score(true_label_list, pred_label_list, average='weighted')


def test(epoch):
    model.eval()
    test_loss = 0

    true_label_list = []
    pred_label_list = []

    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            data = data.to(device)
            labels = labels.to(device)
            recon_batch, pred, hidden = model(data)
            pred_labels = np.argmax(pred.detach().cpu().numpy(), axis=1)

            loss = reconstruction_loss(recon_batch, data) + \
                classification_loss(pred, labels) + \
                center_loss(hidden, labels)  

            test_loss += loss.item()
            true_label_list += list(labels.cpu().numpy())
            pred_label_list += list(pred_labels)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}\tAccuracy: {:.4f}\tF1: {:.4f}'.format(test_loss,
                   accuracy_score(true_label_list, pred_label_list),
            f1_score(true_label_list, pred_label_list, average='weighted')
    ))

    return accuracy_score(true_label_list, pred_label_list), f1_score(true_label_list, pred_label_list, average='weighted')

        
    

best_val_f1 = 0.0
best_train_f1 = 0.0
best_train_acc = 0.0
best_val_acc = 0.0
best_epoch = 0

# Training
for epoch in range(1, num_epochs+1):

    train_acc, train_f1 = train(epoch)
    val_acc, val_f1 = val(epoch)

    if val_f1 >= best_val_f1:
        best_epoch = epoch
        best_val_f1 = val_f1
        best_train_f1 = train_f1
        best_train_acc = train_acc
        best_val_acc = val_acc
        torch.save(model.state_dict(), "./model_epoch"%i+str(epoch)+".pt")


print('==========================================================')
print('===> [Epoch{}] Best Train set Accuracy: {:.4f}\tF1: {:.4f}'.format(best_epoch,
                                                                          best_train_acc,
                                                                         best_train_f1))
print('===> [Epoch{}] Best Validata set Accuracy: {:.4f}\tF1: {:.4f}'.format(best_epoch,
                                                                          best_val_acc,
                  
                                                                             best_val_f1))
   
    
# Test
model = AE().to(device)
model.load_state_dict(torch.load("./model_epoch"%i+str(best_best_epoch)+".pt"))
test_acc, test_f1 = test(epoch)
print('test acc: {}, test f1: {}'.format(test_acc, test_f1))
    
