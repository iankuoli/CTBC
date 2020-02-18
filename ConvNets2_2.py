import networkx as nx
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import numpy as np
import math
import pickle
from tqdm import tqdm_notebook
from multiprocessing import Pool

np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

import xgboost as xgb
from xgboost.sklearn import XGBClassifier



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import datasets, transforms



torch.cuda.set_device(2)



## Model Declaration



class ConvNet(nn.Module):
    def __init__(self, in_dim=256, out_dim=1):
        super(ConvNet, self).__init__()
        
        self.in_dim = in_dim
        self.outdim_en1 = in_dim
        self.outdim_en2 = math.ceil(self.outdim_en1 / 2)
        self.out_dim = out_dim
        
        self.model_conv = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=in_dim*2, kernel_size=2),
            nn.BatchNorm1d(in_dim*2),
            nn.ReLU(),
            nn.Conv1d(in_channels=in_dim*2, out_channels=in_dim*4, kernel_size=2),
            nn.BatchNorm1d(in_dim*4),
            nn.ReLU(),
        )
        
        self.model_fc = nn.Sequential(
            nn.Linear(in_features=self.in_dim*4, out_features=self.outdim_en1),
            nn.BatchNorm1d(self.outdim_en1),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(in_features=self.outdim_en1, out_features=self.outdim_en2),
            nn.BatchNorm1d(self.outdim_en2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=self.outdim_en2, out_features=self.out_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.model_conv(x)
        return self.model_fc(x.view(-1, self.in_dim*4))



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.01, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
    
    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduce:
            return torch.mean(F_loss)
        else:
            F_loss



class FocalLoss2(nn.Module):
    def __init__(self, alpha=0.01, gamma_pos=3, gamma_neg=2, logits=False, reduce=True):
        super(FocalLoss2, self).__init__()
        self.alpha = alpha
        self.gamma_pos=3
        self.gamma_neg=2
        self.logits = logits
        self.reduce = reduce
    
    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        gamma_diff = self.gamma_pos - self.gamma_neg
        F_loss_pos = self.alpha * targets * (1-pt)**self.gamma_pos * BCE_loss
        F_loss_pos = torch.mean(pt)**(-gamma_diff) * F_loss_pos
        F_loss_neg = self.alpha * (1 - targets) * (1-pt)**self.gamma_neg * BCE_loss
        F_loss = F_loss_pos + F_loss_neg
        
        avg_F_loss_pos = torch.sum(F_loss_pos) / torch.sum(targets)
        avg_F_loss_neg = torch.sum(F_loss_neg) / torch.sum(1-targets)
        
        if self.reduce:
            return torch.mean(F_loss), avg_F_loss_pos, avg_F_loss_neg
        else:
            return F_loss, F_loss_pos, F_loss_neg



## Parameters Settings




#
# GRU
# ---------------------
## focal loss
alpha = 1e-4
gamma = 2
gamma_pos = 4
gamma_neg = 2
learn_rate = 1e-4

train_batch_size = 128
test_batch_size = 256

max_epochs = 100




## Data Preparation




data = np.load('GRUArray_and_label_for_NewEmbedding_heter_superv_recur_focal_logisticMF.npz', allow_pickle=True)

GPUArray = data['arr_0']
label = data['arr_1']

GPUArray = GPUArray[-1033905:,:,:]
label = label[-1033905:]

X_train, X_test, y_train, y_test = train_test_split(GPUArray, label, random_state=42)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

train_data = []
for i in range(len(X_train)):
    train_data.append((X_train[i], y_train[i]))
    
test_data = []
for i in range(len(X_test)):
    test_data.append((X_test[i], y_test[i]))

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=test_batch_size)



classifier = ConvNet(in_dim=X_train.shape[2], out_dim=1).cuda()
focal_loss = FocalLoss2(alpha, gamma_pos, gamma_neg)
optim_clsfr = optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), 
                         lr=learn_rate)



def train(epoch, dataloader):
    label_list = []
    pred_y_list = []
    
    clsf_loss_batch = []
    clsf_loss_pos_batch = []
    clsf_loss_neg_batch = []
    for batch_idx, (data, target) in enumerate(dataloader):
        if data.size()[0] != dataloader.batch_size:
            continue
        data, target = Variable(data.cuda()), Variable(target.cuda())
         
        # Update classifier
        optim_clsfr.zero_grad()
        
        pred_y = classifier(data.permute(0, 2, 1)).squeeze(-1)
        clsf_loss, clsf_loss_pos, clsf_loss_neg = focal_loss(pred_y, target)
        clsf_loss.backward()
        optim_clsfr.step()

        clsf_loss_batch.append(clsf_loss)
        if torch.sum(target) > 0:
            clsf_loss_pos_batch.append(clsf_loss_pos)
        clsf_loss_neg_batch.append(clsf_loss_neg)
        
        label_list += list(target.cpu().detach().numpy())
        pred_y_list += list(pred_y.cpu().detach().numpy())
        
        if batch_idx % 2000 == 0:
            print('  Idx {} => clsf: {}'.format(batch_idx, clsf_loss))
    
    clsf_loss_avg = sum(clsf_loss_batch) / len(clsf_loss_batch)
    clsf_loss_pos_avg = sum(clsf_loss_pos_batch) / len(clsf_loss_pos_batch)
    clsf_loss_neg_avg = sum(clsf_loss_neg_batch) / len(clsf_loss_neg_batch)
    
    return np.array(label_list), np.array(pred_y_list), clsf_loss_avg, clsf_loss_pos_avg, clsf_loss_neg_avg



def infer(dataloader):
    label_list = []
    pred_y_list = []   
    
    clsf_loss_batch = []
    clsf_loss_pos_batch = []
    clsf_loss_neg_batch = []
    for batch_idx, (data, target) in enumerate(dataloader):
        if data.size()[0] != dataloader.batch_size:
            continue
        data, target = Variable(data.cuda()), Variable(target.cuda())
         
        # Update classifier       
        pred_y = classifier(data.permute(0, 2, 1)).squeeze(-1)
        clsf_loss, clsf_loss_pos, clsf_loss_neg = focal_loss(pred_y, target)
        
        clsf_loss_batch.append(clsf_loss)
        if torch.sum(target) > 0:
            clsf_loss_pos_batch.append(clsf_loss_pos)
        clsf_loss_neg_batch.append(clsf_loss_neg)
        
        label_list += list(target.cpu().detach().numpy())
        pred_y_list += list(pred_y.cpu().detach().numpy())
    
    clsf_loss_avg = sum(clsf_loss_batch) / len(clsf_loss_batch)
    clsf_loss_pos_avg = sum(clsf_loss_pos_batch) / len(clsf_loss_pos_batch)
    clsf_loss_neg_avg = sum(clsf_loss_neg_batch) / len(clsf_loss_neg_batch)
    
    return np.array(label_list), np.array(pred_y_list), clsf_loss_avg, clsf_loss_pos_avg, clsf_loss_neg_avg



def evaluate(y_true, y_pred):
    prec = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return prec, recall, f1



train_history_loss = []
train_history_auc = []
max_thres = 0.
max_train_auc = 0.

for epoch in range(max_epochs):
    print('Epoch {} -------------------------------------------------------------------------'.format(epoch))
    
    classifier.train()
    label_train, pred_y_train, clsf_loss_train, clsf_loss_pos_train, clsf_loss_neg_train = train(epoch, train_dataloader)
    
    auc_train = roc_auc_score(label_train, pred_y_train)
    train_history_loss.append(clsf_loss_train)
    train_history_auc.append(auc_train)
    
    print('Training => auc: {:.6f}, clsf_pos: {}, clsf_neg: {}'.
          format(auc_train, clsf_loss_pos_train, clsf_loss_neg_train))
    
    if epoch % 1 == 0:
        #
        # Testing
        # ------------------------------------------------------------------------------------        
        thres = np.min(pred_y_train[label_train==1])
        print("Threshold is set to {}".format(thres))
        
        with torch.no_grad():
            classifier.eval()
            label_test, pred_y_test, clsf_loss_test, clsf_loss_pos_test, clsf_loss_neg_test = infer(test_dataloader)    
        
        auc = roc_auc_score(label_test, pred_y_test)
        
        print("Min. Probailities on test set with label 1: {}".format(np.min(pred_y_test[label_test==1])))
        y_predict_bin = np.array(pred_y_test > thres, dtype=int)
        prec, recall, f1 = evaluate(label_test, y_predict_bin)
                
        print('Testing ==> auc: {:.6f}, prec: {:.4f}, rec: {:.4f}, F1score: {:.4f}, clsf_loss: {}'.
              format(auc, prec, recall, f1, clsf_loss_test))
        
        if auc_train > max_train_auc or thres > max_thres:
            max_train_auc = auc_train if auc_train > max_train_auc else max_train_auc
            max_thres = thres if thres > max_thres else max_thres
            torch.save({'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optim_clsfr.state_dict(),
                'loss': focal_loss,
                }, 
               'saved_models/conv1d2_heter_clsfr-auc{:.6f}-thres{:.4f}'.format(auc_train, thres))





