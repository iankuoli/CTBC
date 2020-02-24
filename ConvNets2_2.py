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
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
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


'''
Epoch 0 -------------------------------------------------------------------------
  Idx 0 => clsf: 3.180097337462939e-05
  Idx 2000 => clsf: 4.1567297159872396e-08
  Idx 4000 => clsf: 9.181596816176807e-09
  Idx 6000 => clsf: 3.0312852228320253e-09
Training => auc: 0.984012, clsf_pos: 1.4177763659972697e-05, clsf_neg: 3.927018497051904e-07
Threshold is set to 0.042069803923368454
Min. Probailities on test set with label 1: 0.017674291506409645
Testing ==> auc: 0.974674, prec: 0.0035, rec: 0.9767, F1score: 0.0070, clsf_loss: 6.776513572503973e-09
Epoch 1 -------------------------------------------------------------------------
  Idx 0 => clsf: 2.473788951462552e-09
  Idx 2000 => clsf: 1.0929490645850137e-09
  Idx 4000 => clsf: 5.3109711073418e-10
  Idx 6000 => clsf: 7.051710881889051e-10
Training => auc: 0.986544, clsf_pos: 9.21737046155613e-06, clsf_neg: 1.228757096072286e-09
Threshold is set to 0.004770103842020035
Min. Probailities on test set with label 1: 0.057617560029029846
Testing ==> auc: 0.999823, prec: 0.0002, rec: 1.0000, F1score: 0.0003, clsf_loss: 2.8976829824500783e-09
Epoch 2 -------------------------------------------------------------------------
  Idx 0 => clsf: 7.819139224984895e-10
  Idx 2000 => clsf: 2.2328056070719526e-10
  Idx 4000 => clsf: 2.0501557396190861e-10
  Idx 6000 => clsf: 1.5500294103798495e-10
Training => auc: 0.987420, clsf_pos: 1.0768956599349622e-05, clsf_neg: 5.49017553641562e-10
Threshold is set to 0.00422847643494606
Min. Probailities on test set with label 1: 0.0124081801623106
Testing ==> auc: 0.999467, prec: 0.0002, rec: 1.0000, F1score: 0.0003, clsf_loss: 3.640037604668578e-09
Epoch 3 -------------------------------------------------------------------------
  Idx 0 => clsf: 1.2664903847880993e-10
  Idx 2000 => clsf: 1.9871397582971184e-10
  Idx 4000 => clsf: 7.369310023319642e-11
  Idx 6000 => clsf: 8.815074364898479e-11
Training => auc: 0.995489, clsf_pos: 8.526847523171455e-06, clsf_neg: 2.6755878068662753e-10
Threshold is set to 0.007625599857419729
Min. Probailities on test set with label 1: 0.01630779355764389
Testing ==> auc: 0.999787, prec: 0.0011, rec: 1.0000, F1score: 0.0022, clsf_loss: 3.6711322870530694e-09
Epoch 4 -------------------------------------------------------------------------
  Idx 0 => clsf: 1.2046630359918709e-10
  Idx 2000 => clsf: 1.047521722141731e-10
  Idx 4000 => clsf: 6.175711331213307e-11
  Idx 6000 => clsf: 6.46883102639606e-11
Training => auc: 0.998356, clsf_pos: 6.4510040829190984e-06, clsf_neg: 3.974020557073743e-10
Threshold is set to 0.009090877138078213
Min. Probailities on test set with label 1: 0.0392480306327343
Testing ==> auc: 0.999934, prec: 0.0095, rec: 1.0000, F1score: 0.0189, clsf_loss: 1.82703496776071e-09
Epoch 5 -------------------------------------------------------------------------
  Idx 0 => clsf: 4.252895258183287e-11
  Idx 2000 => clsf: 4.198571698643683e-11
  Idx 4000 => clsf: 6.35648755853424e-11
  Idx 6000 => clsf: 3.4695524231409536e-11
Training => auc: 0.999946, clsf_pos: 5.7035872487176675e-06, clsf_neg: 1.7216429670785516e-10
Threshold is set to 0.02147645317018032
Min. Probailities on test set with label 1: 0.042940910905599594
Testing ==> auc: 0.999966, prec: 0.0453, rec: 1.0000, F1score: 0.0866, clsf_loss: 3.120794955790984e-09
Epoch 6 -------------------------------------------------------------------------
  Idx 0 => clsf: 5.530596813851929e-11
  Idx 2000 => clsf: 4.3120944315244714e-11
  Idx 4000 => clsf: 2.674554501480575e-11
  Idx 6000 => clsf: 2.2346622330360333e-09
Training => auc: 0.999617, clsf_pos: 7.104088581399992e-06, clsf_neg: 1.7187742895607983e-10
Threshold is set to 0.011965308338403702
Min. Probailities on test set with label 1: 0.07618094235658646
Testing ==> auc: 0.999970, prec: 0.0187, rec: 1.0000, F1score: 0.0367, clsf_loss: 2.3029553819498005e-09
Epoch 7 -------------------------------------------------------------------------
  Idx 0 => clsf: 2.7237417876690984e-10
  Idx 2000 => clsf: 7.428187925873075e-11
  Idx 4000 => clsf: 3.724244690417322e-09
  Idx 6000 => clsf: 2.4350493843527943e-10
Training => auc: 0.997927, clsf_pos: 5.63932644581655e-06, clsf_neg: 1.7748567893161038e-10
Threshold is set to 0.006347938906401396
Min. Probailities on test set with label 1: 0.055862341076135635
Testing ==> auc: 0.999936, prec: 0.0053, rec: 1.0000, F1score: 0.0105, clsf_loss: 2.139596722017245e-09
Epoch 8 -------------------------------------------------------------------------
  Idx 0 => clsf: 2.6747265513549223e-10
  Idx 2000 => clsf: 3.6152091098529127e-08
  Idx 4000 => clsf: 1.7938869140143865e-11
  Idx 6000 => clsf: 5.644702760765341e-11
Training => auc: 0.999985, clsf_pos: 4.964941126672784e-06, clsf_neg: 3.1524913235436713e-10
Threshold is set to 0.036402639001607895
Min. Probailities on test set with label 1: 0.03797255456447601
Testing ==> auc: 0.999935, prec: 0.0498, rec: 1.0000, F1score: 0.0949, clsf_loss: 2.439758173267137e-09
Epoch 9 -------------------------------------------------------------------------
  Idx 0 => clsf: 9.991685256949268e-11
  Idx 2000 => clsf: 8.736086853922131e-11
  Idx 4000 => clsf: 2.557553058224471e-10
  Idx 6000 => clsf: 2.573189057664127e-11
Training => auc: 0.999975, clsf_pos: 5.078523827251047e-06, clsf_neg: 1.651299930127692e-10
Threshold is set to 0.026118585839867592
Min. Probailities on test set with label 1: 0.011123161762952805
Testing ==> auc: 0.999814, prec: 0.4516, rec: 0.9767, F1score: 0.6176, clsf_loss: 3.880967103242483e-09
Epoch 10 -------------------------------------------------------------------------
  Idx 0 => clsf: 1.752840060598171e-11
  Idx 2000 => clsf: 2.796759698830975e-11
  Idx 4000 => clsf: 2.315705960320713e-11
  Idx 6000 => clsf: 1.8534666856862003e-11
Training => auc: 0.999944, clsf_pos: 5.029150997870602e-06, clsf_neg: 1.586374087647613e-10
Threshold is set to 0.016078852117061615
Min. Probailities on test set with label 1: 0.07293397188186646
Testing ==> auc: 0.999937, prec: 0.0372, rec: 1.0000, F1score: 0.0717, clsf_loss: 2.0397372679781256e-09
Epoch 11 -------------------------------------------------------------------------
  Idx 0 => clsf: 1.9159071898422475e-11
  Idx 2000 => clsf: 7.770750015678729e-11
  Idx 4000 => clsf: 1.428219088134286e-11
  Idx 6000 => clsf: 2.0972025505106018e-10
Training => auc: 0.999985, clsf_pos: 3.6924909636582015e-06, clsf_neg: 1.3725894032479147e-10
Threshold is set to 0.028964009135961533
Min. Probailities on test set with label 1: 0.04810567572712898
Testing ==> auc: 0.999932, prec: 0.0426, rec: 1.0000, F1score: 0.0817, clsf_loss: 2.621581618456048e-09
Epoch 12 -------------------------------------------------------------------------
  Idx 0 => clsf: 1.6186394238837387e-11
  Idx 2000 => clsf: 2.3966533618802188e-11
  Idx 4000 => clsf: 1.9677308393806214e-11
  Idx 6000 => clsf: 3.2605609184832574e-11
Training => auc: 0.999998, clsf_pos: 2.456007905493607e-06, clsf_neg: 1.4827362948555134e-10
Threshold is set to 0.08942635357379913
Min. Probailities on test set with label 1: 0.010869729332625866
Testing ==> auc: 0.999902, prec: 0.8889, rec: 0.9302, F1score: 0.9091, clsf_loss: 3.758317657087673e-09
Epoch 13 -------------------------------------------------------------------------
  Idx 0 => clsf: 1.9662832126399188e-11
  Idx 2000 => clsf: 2.6728515234442085e-11
  Idx 4000 => clsf: 1.0870407872454191e-11
  Idx 6000 => clsf: 1.0176811650330908e-11
Training => auc: 0.999869, clsf_pos: 4.76427794637857e-06, clsf_neg: 1.0161227415039775e-10
Threshold is set to 0.012571227736771107
Min. Probailities on test set with label 1: 0.06783907860517502
Testing ==> auc: 0.999939, prec: 0.0325, rec: 1.0000, F1score: 0.0630, clsf_loss: 2.882978078488918e-09
Epoch 14 -------------------------------------------------------------------------
  Idx 0 => clsf: 1.4055852835814786e-11
  Idx 2000 => clsf: 4.490695593162286e-10
  Idx 4000 => clsf: 3.6349090404286244e-10
  Idx 6000 => clsf: 1.1976087656295764e-11
Training => auc: 0.999992, clsf_pos: 2.9908405849710107e-06, clsf_neg: 1.927581844141102e-10
Threshold is set to 0.05567270889878273
Min. Probailities on test set with label 1: 0.019926929846405983
Testing ==> auc: 0.999919, prec: 0.1405, rec: 0.9767, F1score: 0.2456, clsf_loss: 3.2658402648877427e-09
Epoch 15 -------------------------------------------------------------------------
  Idx 0 => clsf: 1.2783750620581902e-11
  Idx 2000 => clsf: 1.615479972016942e-11
  Idx 4000 => clsf: 4.387665925031925e-11
  Idx 6000 => clsf: 1.1835641841595468e-11
Training => auc: 0.999991, clsf_pos: 2.8810395633627195e-06, clsf_neg: 1.0705120123688516e-10
Threshold is set to 0.03907858580350876
Min. Probailities on test set with label 1: 0.019675686955451965
Testing ==> auc: 0.999922, prec: 0.0733, rec: 0.9767, F1score: 0.1364, clsf_loss: 3.1810547529431688e-09
Epoch 16 -------------------------------------------------------------------------
  Idx 0 => clsf: 2.623142675295398e-11
  Idx 2000 => clsf: 5.304361117008938e-11
  Idx 4000 => clsf: 8.207490242995163e-12
  Idx 6000 => clsf: 9.826322047712388e-12
Training => auc: 0.999998, clsf_pos: 2.1857701995031675e-06, clsf_neg: 7.998341572390544e-11
Threshold is set to 0.06598863750696182
Min. Probailities on test set with label 1: 0.003597858129069209
Testing ==> auc: 0.999525, prec: 0.9091, rec: 0.9302, F1score: 0.9195, clsf_loss: 5.606712871752961e-09
Epoch 17 -------------------------------------------------------------------------
  Idx 0 => clsf: 1.1762273446902505e-11
  Idx 2000 => clsf: 9.192134033109145e-12
  Idx 4000 => clsf: 9.19030389984199e-12
  Idx 6000 => clsf: 1.9899106667997657e-11
Training => auc: 0.999975, clsf_pos: 3.4574179608171107e-06, clsf_neg: 1.8256371414615558e-10
Threshold is set to 0.01627841591835022
Min. Probailities on test set with label 1: 0.005854449234902859
Testing ==> auc: 0.999678, prec: 0.0840, rec: 0.9767, F1score: 0.1547, clsf_loss: 4.9219099906849806e-09
Epoch 18 -------------------------------------------------------------------------
  Idx 0 => clsf: 2.2994200293835476e-11
  Idx 2000 => clsf: 3.798718839487236e-10
  Idx 4000 => clsf: 7.848527383558235e-12
  Idx 6000 => clsf: 1.2480525293789846e-11
Training => auc: 0.999994, clsf_pos: 3.0306803182611475e-06, clsf_neg: 1.2851915365263977e-10
Threshold is set to 0.057559721171855927
Min. Probailities on test set with label 1: 0.0342058427631855
Testing ==> auc: 0.999931, prec: 0.3590, rec: 0.9767, F1score: 0.5250, clsf_loss: 3.5859999414356025e-09
Epoch 19 -------------------------------------------------------------------------
  Idx 0 => clsf: 2.1808053626837243e-11
  Idx 2000 => clsf: 1.1375086636511433e-11
  Idx 4000 => clsf: 1.806585783747927e-11
  Idx 6000 => clsf: 1.7897584803083788e-10
Training => auc: 0.999990, clsf_pos: 3.134448661512579e-06, clsf_neg: 1.2103909541316682e-10
Threshold is set to 0.034912291914224625
Min. Probailities on test set with label 1: 0.005457509309053421
Testing ==> auc: 0.999842, prec: 0.1843, rec: 0.9302, F1score: 0.3077, clsf_loss: 5.072182673870884e-09
Epoch 20 -------------------------------------------------------------------------
  Idx 0 => clsf: 1.3696956763231682e-11
  Idx 2000 => clsf: 4.746208981387667e-10
  Idx 4000 => clsf: 1.243938459183358e-11
  Idx 6000 => clsf: 5.273234973679486e-12
Training => auc: 0.999997, clsf_pos: 2.2446158709499286e-06, clsf_neg: 5.727815791112256e-11
Threshold is set to 0.05452180653810501
Min. Probailities on test set with label 1: 0.003080059075728059
Testing ==> auc: 0.996904, prec: 0.8163, rec: 0.9302, F1score: 0.8696, clsf_loss: 5.4293605167288206e-09
Epoch 21 -------------------------------------------------------------------------
  Idx 0 => clsf: 7.842456718754054e-12
  Idx 2000 => clsf: 1.1155242528315679e-11
  Idx 4000 => clsf: 4.9478988614626296e-12
  Idx 6000 => clsf: 8.654909254557364e-12
Training => auc: 0.999985, clsf_pos: 3.848815595119959e-06, clsf_neg: 1.5159962174493558e-10
Threshold is set to 0.020875653252005577
Min. Probailities on test set with label 1: 0.023123309016227722
Testing ==> auc: 0.999934, prec: 0.0570, rec: 1.0000, F1score: 0.1078, clsf_loss: 3.3928269083105533e-09
Epoch 22 -------------------------------------------------------------------------
  Idx 0 => clsf: 1.5345659465371142e-10
  Idx 2000 => clsf: 5.501275650299231e-12
  Idx 4000 => clsf: 9.348277360543555e-12
  Idx 6000 => clsf: 9.292738453736682e-12
Training => auc: 0.999989, clsf_pos: 2.7361597858543973e-06, clsf_neg: 7.292741410758197e-11
Threshold is set to 0.029038527980446815
Min. Probailities on test set with label 1: 0.00556844100356102
Testing ==> auc: 0.999881, prec: 0.1522, rec: 0.9767, F1score: 0.2633, clsf_loss: 4.3210821587535975e-09
Epoch 23 -------------------------------------------------------------------------
  Idx 0 => clsf: 6.914073046732083e-12
  Idx 2000 => clsf: 9.990107699420214e-12
  Idx 4000 => clsf: 1.1983554773498106e-11
  Idx 6000 => clsf: 1.1166637614579145e-09
Training => auc: 0.999968, clsf_pos: 3.7955039715598105e-06, clsf_neg: 1.330833082624494e-10
Threshold is set to 0.014160431921482086
Min. Probailities on test set with label 1: 0.06285425275564194
Testing ==> auc: 0.999897, prec: 0.0393, rec: 1.0000, F1score: 0.0757, clsf_loss: 2.552049460646799e-09
Epoch 24 -------------------------------------------------------------------------
  Idx 0 => clsf: 9.16762759456402e-12
  Idx 2000 => clsf: 2.9319695976637306e-11
  Idx 4000 => clsf: 5.73566151171323e-12
  Idx 6000 => clsf: 7.502823015648197e-12
Training => auc: 0.999987, clsf_pos: 3.128912567262887e-06, clsf_neg: 1.6404451408380538e-10
Threshold is set to 0.039402686059474945
Min. Probailities on test set with label 1: 0.03392123058438301
Testing ==> auc: 0.999922, prec: 0.0649, rec: 0.9767, F1score: 0.1217, clsf_loss: 3.1273605927140125e-09
Epoch 25 -------------------------------------------------------------------------
  Idx 0 => clsf: 5.050874649081827e-12
  Idx 2000 => clsf: 5.812217027112432e-12
  Idx 4000 => clsf: 4.785491447556467e-12
  Idx 6000 => clsf: 4.921901861770772e-12
Training => auc: 0.999991, clsf_pos: 2.1945190837868722e-06, clsf_neg: 9.22651954837761e-11
Threshold is set to 0.04655322805047035
Min. Probailities on test set with label 1: 0.006852686870843172
Testing ==> auc: 0.999900, prec: 0.1695, rec: 0.9302, F1score: 0.2867, clsf_loss: 4.127334918280212e-09
Epoch 26 -------------------------------------------------------------------------
  Idx 0 => clsf: 2.953482458600831e-10
  Idx 2000 => clsf: 5.5948119401239005e-12
  Idx 4000 => clsf: 5.517459319287488e-12
  Idx 6000 => clsf: 4.7847706699521986e-12
Training => auc: 0.999992, clsf_pos: 3.426252987992484e-06, clsf_neg: 1.2722446707247315e-10
Threshold is set to 0.05697764456272125
Min. Probailities on test set with label 1: 0.04139380529522896
Testing ==> auc: 0.999872, prec: 0.1449, rec: 0.9302, F1score: 0.2508, clsf_loss: 3.327045527967698e-09
Epoch 27 -------------------------------------------------------------------------
  Idx 0 => clsf: 4.2121419199792065e-12
  Idx 2000 => clsf: 5.48666754390803e-12
  Idx 4000 => clsf: 6.069302178890457e-12
  Idx 6000 => clsf: 2.6801724901936996e-12
Training => auc: 0.999995, clsf_pos: 2.593475983303506e-06, clsf_neg: 1.0941256234353602e-10
Threshold is set to 0.05497302860021591
Min. Probailities on test set with label 1: 0.009865447878837585
Testing ==> auc: 0.999829, prec: 0.1717, rec: 0.9302, F1score: 0.2899, clsf_loss: 4.911965056919598e-09
Epoch 28 -------------------------------------------------------------------------
  Idx 0 => clsf: 5.702323162271039e-12
  Idx 2000 => clsf: 6.707787103543694e-12
  Idx 4000 => clsf: 2.6709374295608157e-11
  Idx 6000 => clsf: 4.207073578399445e-11
Training => auc: 0.999978, clsf_pos: 3.416762183405808e-06, clsf_neg: 1.0317076359900312e-10
Threshold is set to 0.018782800063490868
Min. Probailities on test set with label 1: 0.02162165194749832
Testing ==> auc: 0.999902, prec: 0.0590, rec: 1.0000, F1score: 0.1114, clsf_loss: 3.4409597393647573e-09
Epoch 29 -------------------------------------------------------------------------
  Idx 0 => clsf: 8.777535642767731e-11
  Idx 2000 => clsf: 3.4373552615374336e-11
  Idx 4000 => clsf: 6.3479985157322e-11
  Idx 6000 => clsf: 5.350741551224392e-12
Training => auc: 0.999994, clsf_pos: 2.5309675493190298e-06, clsf_neg: 7.236103383156944e-11
Threshold is set to 0.04678452014923096
Min. Probailities on test set with label 1: 0.008769948035478592
Testing ==> auc: 0.999897, prec: 0.1591, rec: 0.9767, F1score: 0.2736, clsf_loss: 3.988388730391534e-09
Epoch 30 -------------------------------------------------------------------------
  Idx 0 => clsf: 2.7339825282163277e-12
  Idx 2000 => clsf: 4.078090296011361e-12
  Idx 4000 => clsf: 5.958975934222677e-12
  Idx 6000 => clsf: 2.0843914701890176e-12
Training => auc: 0.999997, clsf_pos: 2.2534729851031443e-06, clsf_neg: 1.3964437939328889e-10
Threshold is set to 0.07935924082994461
Min. Probailities on test set with label 1: 0.006430990528315306
Testing ==> auc: 0.999862, prec: 0.7273, rec: 0.9302, F1score: 0.8163, clsf_loss: 4.7155950255728385e-09
Epoch 31 -------------------------------------------------------------------------
  Idx 0 => clsf: 1.7688365962914565e-12
  Idx 2000 => clsf: 2.9690527380416e-12
  Idx 4000 => clsf: 4.252486904277042e-11
  Idx 6000 => clsf: 2.8031249196813768e-12
Training => auc: 0.999993, clsf_pos: 2.3058580609358614e-06, clsf_neg: 4.7196076907729534e-11
Threshold is set to 0.040091872215270996
Min. Probailities on test set with label 1: 0.010271180421113968
Testing ==> auc: 0.999853, prec: 0.0820, rec: 0.9302, F1score: 0.1507, clsf_loss: 4.219815608053068e-09
Epoch 32 -------------------------------------------------------------------------
  Idx 0 => clsf: 2.6215791690265e-12
  Idx 2000 => clsf: 3.462985757526904e-12
  Idx 4000 => clsf: 8.699985870608273e-11
  Idx 6000 => clsf: 1.5569846844101787e-12
Training => auc: 0.999996, clsf_pos: 2.2348228867485886e-06, clsf_neg: 1.07841194307845e-10
Threshold is set to 0.06484325975179672
Min. Probailities on test set with label 1: 0.0033310684375464916
Testing ==> auc: 0.999437, prec: 0.1961, rec: 0.9302, F1score: 0.3239, clsf_loss: 5.115790902010531e-09
Epoch 33 -------------------------------------------------------------------------
  Idx 0 => clsf: 7.833456973360686e-12
  Idx 2000 => clsf: 2.7143396419404553e-09
  Idx 4000 => clsf: 2.8771134780170016e-12
  Idx 6000 => clsf: 2.0916504205742426e-12
Training => auc: 0.999992, clsf_pos: 2.9837074180250056e-06, clsf_neg: 8.617891122941757e-11
Threshold is set to 0.041820891201496124
Min. Probailities on test set with label 1: 0.007536693941801786
Testing ==> auc: 0.999825, prec: 0.1036, rec: 0.9302, F1score: 0.1865, clsf_loss: 5.3403890198922e-09
Epoch 34 -------------------------------------------------------------------------
  Idx 0 => clsf: 7.480131097858944e-11
  Idx 2000 => clsf: 3.3165540626323153e-12
  Idx 4000 => clsf: 2.702665782144953e-12
  Idx 6000 => clsf: 9.562286726327862e-12
Training => auc: 0.999999, clsf_pos: 1.7759648471837863e-06, clsf_neg: 9.133163669794442e-11
Threshold is set to 0.09692412614822388
Min. Probailities on test set with label 1: 0.0043848794884979725
Testing ==> auc: 0.999795, prec: 0.7018, rec: 0.9302, F1score: 0.8000, clsf_loss: 5.972694339106965e-09
Epoch 35 -------------------------------------------------------------------------
  Idx 0 => clsf: 4.086367529076984e-12
  Idx 2000 => clsf: 2.499957706125766e-11
  Idx 4000 => clsf: 5.429280358626443e-10
  Idx 6000 => clsf: 5.2497672875517765e-11
Training => auc: 0.999993, clsf_pos: 2.2145218281366397e-06, clsf_neg: 5.818910978061531e-11
Threshold is set to 0.0451289638876915
Min. Probailities on test set with label 1: 0.008911632001399994
Testing ==> auc: 0.999830, prec: 0.1533, rec: 0.9302, F1score: 0.2632, clsf_loss: 5.310671902236663e-09
Epoch 36 -------------------------------------------------------------------------
  Idx 0 => clsf: 1.2235316926290096e-10
  Idx 2000 => clsf: 4.6513400769887525e-12
  Idx 4000 => clsf: 2.4154343889609686e-12
  Idx 6000 => clsf: 1.5250466904939697e-12
Training => auc: 0.999998, clsf_pos: 1.7965813867704128e-06, clsf_neg: 4.913838086428868e-11
Threshold is set to 0.06879152357578278
Min. Probailities on test set with label 1: 0.001360047492198646
Testing ==> auc: 0.978942, prec: 0.9091, rec: 0.9302, F1score: 0.9195, clsf_loss: 6.827893805905205e-09
Epoch 37 -------------------------------------------------------------------------
  Idx 0 => clsf: 2.9519321015358813e-12
  Idx 2000 => clsf: 8.933629010166033e-12
  Idx 4000 => clsf: 2.0031190248182007e-12
  Idx 6000 => clsf: 1.4920726329817335e-12
Training => auc: 0.999999, clsf_pos: 1.651143179515202e-06, clsf_neg: 5.3793913767918866e-11
Threshold is set to 0.08773916959762573
Min. Probailities on test set with label 1: 0.0021049317438155413
Testing ==> auc: 0.998583, prec: 0.9524, rec: 0.9302, F1score: 0.9412, clsf_loss: 6.077982117602687e-09
'''
