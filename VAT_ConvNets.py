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



## Virtual Adversarial Training



import contextlib


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True
            
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)

    
def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VATLoss(nn.Module):

    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):
        with torch.no_grad():
            pred = F.softmax(model(x), dim=1)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model(x + self.xi * d)
                logp_hat = F.log_softmax(pred_hat, dim=1)
                #adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')   # for PyTorch v1.0
                adv_distance = F.kl_div(logp_hat, pred)         # for PyTorch v0.4
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()
    
            # calc LDS
            r_adv = d * self.eps
            pred_hat = model(x + r_adv)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            #lds = F.kl_div(logp_hat, pred, reduction='batchmean')    # for PyTorch v1.0
            lds = F.kl_div(logp_hat, pred)          # for PyTorch v1.0

        return lds



## Parameters Settings


#
# Classifier
# ---------------------
## focal loss
alpha = 1e-4
gamma_pos = 8
gamma_neg = 2
learn_rate = 1e-4

#
# VAT
# ---------------------
vat_xi = 10
vat_eps = 1.0
vat_ip = 1

#
# Training process
# ---------------------
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



vat_loss = VATLoss(xi=vat_xi, eps=vat_eps, ip=vat_ip)



def train(epoch, dataloader):
    label_list = []
    pred_y_list = []
    
    clsf_loss_batch = []
    clsf_loss_pos_batch = []
    clsf_loss_neg_batch = []
    vat_batch = []
    for batch_idx, (data, target) in enumerate(dataloader):
        if data.size()[0] != dataloader.batch_size:
            continue
        data, target = Variable(data.cuda()), Variable(target.cuda())
        data = data.permute(0, 2, 1)
        
        #
        # Update classifier on real samples
        #
        optim_clsfr.zero_grad()
        
        vat_kld = vat_loss(classifier, data)
        
        pred_y = classifier(data).squeeze(-1)
        clsf_loss, clsf_loss_pos, clsf_loss_neg = focal_loss(pred_y, target)
        loss = clsf_loss + vat_kld
        
        loss.backward()
        optim_clsfr.step()
        
        #
        # Record the losses
        #
        vat_batch.append(vat_kld)
        clsf_loss_batch.append(clsf_loss)
        if torch.sum(target) > 0:
            clsf_loss_pos_batch.append(clsf_loss_pos)
        clsf_loss_neg_batch.append(clsf_loss_neg)
        
        label_list += list(target.cpu().detach().numpy())
        pred_y_list += list(pred_y.cpu().detach().numpy())
        
        #if batch_idx % 2000 == 0:
        #    print('  Idx {} => clsf: {}'.format(batch_idx, clsf_loss))
    
    vat_batch = sum(vat_batch) / len(vat_batch)
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


print('Parameter Setting ----------------------------------------------------------------------')
print('Model = VAT + Conv1D')
print('conv1d use activation = {}'.format(True))
print('graph_emdeding = Heter_AutoEncoder')
print('alpha = {}'.format(alpha))
print('gamma_pos = {}'.format(gamma_pos))
print('gamma_neg = {}'.format(gamma_neg))
print('learn_rate = {}'.format(learn_rate))
print('train_batch_size = {}'.format(train_batch_size))
print('test_batch_size = {}'.format(test_batch_size))
print('max_epochs = {}'.format(max_epochs))
print('\n')


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
        print("            Threshold is set to {}".format(thres))
        
        with torch.no_grad():
            classifier.eval()
            label_test, pred_y_test, clsf_loss_test, clsf_loss_pos_test, clsf_loss_neg_test = infer(test_dataloader)    
        
        auc = roc_auc_score(label_test, pred_y_test)
        
        print("            Min. Probailities on test set with label 1: {}".
              format(np.min(pred_y_test[label_test==1])))
        y_predict_bin = np.array(pred_y_test > thres, dtype=int)
        prec, recall, f1 = evaluate(label_test, y_predict_bin)
                
        print('Testing ==> auc: {:.6f}, prec: {:.4f}, rec: {:.4f}, F1score: {:.4f}, clsf_loss: {}'.
              format(auc, prec, recall, f1, clsf_loss_test))
        
#         if auc_train > max_train_auc or thres > max_thres:
#             max_train_auc = auc_train if auc_train > max_train_auc else max_train_auc
#             max_thres = thres if thres > max_thres else max_thres
#             torch.save({'epoch': epoch,
#                 'model_state_dict': classifier.state_dict(),
#                 'optimizer_state_dict': optim_clsfr.state_dict(),
#                 'loss': focal_loss,
#                 }, 
#                'saved_models/conv1d2_heter_clsfr_focal{}{}-auc{:.6f}-thres{:.4f}'.
#                        format(gamma_pos, gamma_neg, auc_train, thres))










Parameter Setting ----------------------------------------------------------------------
Model = VAT + Conv1D
conv1d use activation = True
graph_emdeding = Heter_AutoEncoder
alpha = 0.0001
gamma_pos = 8
gamma_neg = 2
learn_rate = 0.0001
train_batch_size = 128
test_batch_size = 256
max_epochs = 100


Epoch 0 -------------------------------------------------------------------------
Training => auc: 0.986189, clsf_pos: 1.3337967175175436e-05, clsf_neg: 1.9876877388469438e-07
            Threshold is set to 0.021823186427354813
            Min. Probailities on test set with label 1: 0.021828558295965195
Testing ==> auc: 0.976315, prec: 0.0002, rec: 1.0000, F1score: 0.0004, clsf_loss: 5.997119245648719e-09
Epoch 1 -------------------------------------------------------------------------
Training => auc: 0.991309, clsf_pos: 1.0799921255966183e-05, clsf_neg: 1.0730236699174611e-09
            Threshold is set to 0.006082397885620594
            Min. Probailities on test set with label 1: 0.006392541341483593
Testing ==> auc: 0.976730, prec: 0.0002, rec: 1.0000, F1score: 0.0003, clsf_loss: 4.7295483085463275e-09
Epoch 2 -------------------------------------------------------------------------
Training => auc: 0.999075, clsf_pos: 5.455747214000439e-06, clsf_neg: 3.7317371415213074e-10
            Threshold is set to 0.015952017158269882
            Min. Probailities on test set with label 1: 0.011405078694224358
Testing ==> auc: 0.999592, prec: 0.0720, rec: 0.9767, F1score: 0.1342, clsf_loss: 3.718561236709661e-09
Epoch 3 -------------------------------------------------------------------------
Training => auc: 0.995136, clsf_pos: 8.894459824659862e-06, clsf_neg: 3.085455502205292e-10
            Threshold is set to 0.005677458830177784
            Min. Probailities on test set with label 1: 0.020261097699403763
Testing ==> auc: 0.999924, prec: 0.0002, rec: 1.0000, F1score: 0.0004, clsf_loss: 3.470557619067449e-09
Epoch 4 -------------------------------------------------------------------------
Training => auc: 0.999198, clsf_pos: 9.687207239039708e-06, clsf_neg: 2.625478168205575e-10
            Threshold is set to 0.012917846441268921
            Min. Probailities on test set with label 1: 0.010986470617353916
Testing ==> auc: 0.999806, prec: 0.0631, rec: 0.9767, F1score: 0.1185, clsf_loss: 4.132422848357464e-09
Epoch 5 -------------------------------------------------------------------------
Training => auc: 0.999911, clsf_pos: 7.924136298242956e-06, clsf_neg: 2.8169061527805184e-10
            Threshold is set to 0.02172679826617241
            Min. Probailities on test set with label 1: 0.00842991191893816
Testing ==> auc: 0.998372, prec: 0.0512, rec: 0.9302, F1score: 0.0970, clsf_loss: 4.8870849589377485e-09
Epoch 6 -------------------------------------------------------------------------
Training => auc: 0.999198, clsf_pos: 6.748757186869625e-06, clsf_neg: 2.290564127260808e-10
            Threshold is set to 0.009936057962477207
            Min. Probailities on test set with label 1: 0.013297514989972115
Testing ==> auc: 0.999855, prec: 0.0123, rec: 1.0000, F1score: 0.0243, clsf_loss: 4.267045383699042e-09
Epoch 7 -------------------------------------------------------------------------
Training => auc: 0.999975, clsf_pos: 6.799255970690865e-06, clsf_neg: 1.998641807610113e-10
            Threshold is set to 0.03720400482416153
            Min. Probailities on test set with label 1: 0.013799737207591534
Testing ==> auc: 0.999900, prec: 0.6557, rec: 0.9302, F1score: 0.7692, clsf_loss: 4.33659552712129e-09
Epoch 8 -------------------------------------------------------------------------
Training => auc: 0.999871, clsf_pos: 5.6704634516790975e-06, clsf_neg: 2.7227792243067483e-10
            Threshold is set to 0.015154444612562656
            Min. Probailities on test set with label 1: 0.02020658738911152
Testing ==> auc: 0.999902, prec: 0.0429, rec: 1.0000, F1score: 0.0822, clsf_loss: 3.918425139914916e-09
Epoch 9 -------------------------------------------------------------------------
Training => auc: 0.999988, clsf_pos: 5.9544163377722725e-06, clsf_neg: 1.7215093239819623e-10
            Threshold is set to 0.04251478612422943
            Min. Probailities on test set with label 1: 0.1615634709596634
Testing ==> auc: 0.999954, prec: 0.0382, rec: 1.0000, F1score: 0.0736, clsf_loss: 3.555826522116945e-09
Epoch 10 -------------------------------------------------------------------------
Training => auc: 0.999263, clsf_pos: 5.381308710639132e-06, clsf_neg: 1.7922263673142425e-10
            Threshold is set to 0.007887162268161774
            Min. Probailities on test set with label 1: 0.041036128997802734
Testing ==> auc: 0.999936, prec: 0.0186, rec: 1.0000, F1score: 0.0365, clsf_loss: 2.4163910872232464e-09
