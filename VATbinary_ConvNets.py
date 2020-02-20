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




torch.cuda.set_device(1)




class ConvNet(nn.Module):
    def __init__(self, in_dim=256, out_dim=2):
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

    def __init__(self, xi=1e-6, eps=0.1, ip=1):
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


class VATLoss2(nn.Module):

    def __init__(self, xi=1e-6, eps_pos=100, eps_neg=1., ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps_pos: hyperparameter of VAT (default: 100.0)
        :param eps_neg: hyperparameter of VAT (default: 0.1)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss2, self).__init__()
        self.xi = xi
        self.eps_pos = eps_pos
        self.eps_neg = eps_neg
        self.ip = ip

    def forward(self, model, x, y):
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
            r_adv = d * (self.eps_pos * y + self.eps_neg * (1-y)).reshape(-1, 1, 1)
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
vat_xi = 1e-6
vat_eps_pos = 10
vat_eps_neg = 0.1
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




classifier = ConvNet(in_dim=X_train.shape[2], out_dim=2).cuda()
focal_loss = FocalLoss2(alpha, gamma_pos, gamma_neg)
optim_clsfr = optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), 
                         lr=learn_rate)
vat_loss2 = VATLoss2(xi=vat_xi, eps_pos=vat_eps_pos, eps_neg=vat_eps_neg, ip=vat_ip)




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
        tmp = target.reshape(-1, 1)
        onehot_target = torch.cat([1-tmp, tmp], dim=1)
        
        #
        # Update classifier on real samples
        #
        optim_clsfr.zero_grad()
        
        vat_kld = vat_loss2(classifier, data, target)
        
        pred_y = classifier(data).squeeze(-1)
        
        clsf_loss, clsf_loss_pos, clsf_loss_neg = focal_loss(pred_y, onehot_target)
        loss = clsf_loss + vat_kld
        
        loss.backward()
        optim_clsfr.step()
        
        #
        # Record the losses
        #
        pred_yy = torch.softmax(pred_y, dim=1)[:, 1]
        vat_batch.append(vat_kld)
        clsf_loss_batch.append(clsf_loss)
        if torch.sum(target) > 0:
            clsf_loss_pos_batch.append(clsf_loss_pos)
        clsf_loss_neg_batch.append(clsf_loss_neg)
        
        label_list += list(target.cpu().detach().numpy())
        pred_y_list += list(pred_yy.cpu().detach().numpy())
        
        #if batch_idx % 2000 == 0:
        #    print('  Idx {} => clsf: {}'.format(batch_idx, clsf_loss))
    
    vat_loss_avg = sum(vat_batch) / len(vat_batch)
    clsf_loss_avg = sum(clsf_loss_batch) / len(clsf_loss_batch)
    clsf_loss_pos_avg = sum(clsf_loss_pos_batch) / len(clsf_loss_pos_batch)
    clsf_loss_neg_avg = sum(clsf_loss_neg_batch) / len(clsf_loss_neg_batch)
    
    return np.array(label_list), np.array(pred_y_list), clsf_loss_avg, clsf_loss_pos_avg, clsf_loss_neg_avg, vat_loss_avg




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
        pred_y = torch.softmax(pred_y, dim=1)[:, 1]
        
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
print('vat_xi = {}'.format(vat_xi))
print('vat_eps_pos = {}'.format(vat_eps_pos))
print('vat_eps_neg = {}'.format(vat_eps_neg))
print('vat_ip = {}'.format(vat_ip))

print('\n')


for epoch in range(max_epochs):
    print('Epoch {} -------------------------------------------------------------------------'.format(epoch))
    
    classifier.train()
    label_train, pred_y_train, clsf_loss_train, clsf_loss_pos_train, clsf_loss_neg_train, vat_loss_train = train(epoch, train_dataloader)
    
    auc_train = roc_auc_score(label_train, pred_y_train)
    train_history_loss.append(clsf_loss_train)
    train_history_auc.append(auc_train)
    print('Training => auc: {:.6f}, clsf_pos: {}, clsf_neg: {}, vat_loss: {}'.
          format(auc_train, clsf_loss_pos_train, clsf_loss_neg_train, vat_loss_train))
    
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
        
        if auc_train > max_train_auc:
            max_train_auc = auc_train if auc_train > max_train_auc else max_train_auc
            torch.save({'epoch': epoch,
                        'model_state_dict': classifier.state_dict(),
                        'optimizer_state_dict': optim_clsfr.state_dict(),
                        'loss': focal_loss,
                       }, 
                       'saved_models/VATConv_heter_clsfr_xi{}_eps{}{}_focal{}{}_BestAUC'.
                       format(int(-math.log10(vat_xi)), int(-math.log10(vat_eps_pos)), 
                              int(-math.log10(vat_eps_neg)), gamma_pos, gamma_neg))
        if thres > max_thres:
            max_thres = thres if thres > max_thres else max_thres
            torch.save({'epoch': epoch,
                        'model_state_dict': classifier.state_dict(),
                        'optimizer_state_dict': optim_clsfr.state_dict(),
                        'loss': focal_loss,
                       }, 
                       'saved_models/VATConv_heter_clsfr_xi{}_eps{}{}_focal{}{}_BestThres'.
                       format(int(-math.log10(vat_xi)), int(-math.log10(vat_eps_pos)), 
                              int(-math.log10(vat_eps_neg)), gamma_pos, gamma_neg))
        



'''
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
vat_xi = 1e-06
vat_eps_pos = 10
vat_eps_neg = 0.1
vat_ip = 1


Epoch 0 -------------------------------------------------------------------------
Training => auc: 0.430158, clsf_pos: 1.20398781291442e-05, clsf_neg: 8.76358990353765e-06, vat_loss: 1.6026593584683724e-05
            Threshold is set to 0.3398132622241974
            Min. Probailities on test set with label 1: 0.42453762888908386
Testing ==> auc: 0.354335, prec: 0.0002, rec: 1.0000, F1score: 0.0004, clsf_loss: 1.049964612320764e-05
Epoch 1 -------------------------------------------------------------------------
Training => auc: 0.463860, clsf_pos: 4.739088126370916e-06, clsf_neg: 4.511213319347007e-06, vat_loss: 4.87121087644482e-06
            Threshold is set to 0.37438109517097473
            Min. Probailities on test set with label 1: 0.4165689945220947
Testing ==> auc: 0.390801, prec: 0.0002, rec: 1.0000, F1score: 0.0004, clsf_loss: 8.704226274858229e-06
Epoch 2 -------------------------------------------------------------------------
Training => auc: 0.589552, clsf_pos: 4.342293777881423e-06, clsf_neg: 1.4040322184882825e-06, vat_loss: 4.175289177510422e-06
            Threshold is set to 0.33540868759155273
            Min. Probailities on test set with label 1: 0.3395012617111206
Testing ==> auc: 0.544638, prec: 0.0002, rec: 1.0000, F1score: 0.0004, clsf_loss: 7.097728484950494e-06
Epoch 3 -------------------------------------------------------------------------
Training => auc: 0.252225, clsf_pos: 7.192597877292428e-06, clsf_neg: 1.0764204034785507e-06, vat_loss: 2.993034740939038e-06
            Threshold is set to 0.2689416706562042
            Min. Probailities on test set with label 1: 0.269136905670166
Testing ==> auc: 0.136162, prec: 0.0002, rec: 1.0000, F1score: 0.0003, clsf_loss: 5.841909114678856e-06
Epoch 4 -------------------------------------------------------------------------
Training => auc: 0.526386, clsf_pos: 4.5818419494025875e-06, clsf_neg: 8.670681239664191e-08, vat_loss: 1.7969897498915088e-06
            Threshold is set to 0.26918825507164
            Min. Probailities on test set with label 1: 0.333265483379364
Testing ==> auc: 0.999862, prec: 0.0002, rec: 1.0000, F1score: 0.0004, clsf_loss: 4.109113888262073e-06
Epoch 5 -------------------------------------------------------------------------
Training => auc: 0.995613, clsf_pos: 2.6073239496327005e-06, clsf_neg: 9.656503152655205e-09, vat_loss: 5.075125386611035e-07
            Threshold is set to 0.3287423253059387
            Min. Probailities on test set with label 1: 0.32711121439933777
Testing ==> auc: 0.999430, prec: 0.8511, rec: 0.9302, F1score: 0.8889, clsf_loss: 3.852802819892531e-06
Epoch 6 -------------------------------------------------------------------------
Training => auc: 0.995609, clsf_pos: 2.3188003979157656e-06, clsf_neg: 1.0603662836672356e-08, vat_loss: 2.9294457704054366e-07
            Threshold is set to 0.325691282749176
            Min. Probailities on test set with label 1: 0.3226126432418823
Testing ==> auc: 0.999954, prec: 1.0000, rec: 0.9302, F1score: 0.9639, clsf_loss: 3.7283439269231167e-06
Epoch 7 -------------------------------------------------------------------------
Training => auc: 0.998878, clsf_pos: 1.3884480267734034e-06, clsf_neg: 1.1545964184733748e-08, vat_loss: 1.8008850588557834e-07
            Threshold is set to 0.3235968053340912
            Min. Probailities on test set with label 1: 0.3211672604084015
Testing ==> auc: 0.999934, prec: 0.9524, rec: 0.9302, F1score: 0.9412, clsf_loss: 3.719034339155769e-06
Epoch 8 -------------------------------------------------------------------------
Training => auc: 0.995800, clsf_pos: 9.657953796704533e-07, clsf_neg: 1.25355938962457e-08, vat_loss: 1.2161221718542947e-07
            Threshold is set to 0.31894227862358093
            Min. Probailities on test set with label 1: 0.31558847427368164
Testing ==> auc: 0.970809, prec: 1.0000, rec: 0.9302, F1score: 0.9639, clsf_loss: 3.5966643281426514e-06
Epoch 9 -------------------------------------------------------------------------
Training => auc: 0.992411, clsf_pos: 6.921843578311382e-07, clsf_neg: 1.3893264316777731e-08, vat_loss: 8.671356965805899e-08
            Threshold is set to 0.3144271969795227
            Min. Probailities on test set with label 1: 0.3129504919052124
Testing ==> auc: 0.982924, prec: 1.0000, rec: 0.9302, F1score: 0.9639, clsf_loss: 3.4938734643219505e-06
Epoch 10 -------------------------------------------------------------------------
Training => auc: 0.989851, clsf_pos: 5.931280497861735e-07, clsf_neg: 1.5178743595356536e-08, vat_loss: 5.471820330171795e-08
            Threshold is set to 0.3098629415035248
            Min. Probailities on test set with label 1: 0.31041601300239563
Testing ==> auc: 0.994721, prec: 0.0002, rec: 1.0000, F1score: 0.0004, clsf_loss: 3.40817314281594e-06
Epoch 11 -------------------------------------------------------------------------
Training => auc: 0.996994, clsf_pos: 9.207719244841428e-07, clsf_neg: 2.2910899843964216e-08, vat_loss: 1.4778458989894716e-07
            Threshold is set to 0.31079837679862976
            Min. Probailities on test set with label 1: 0.3086114525794983
Testing ==> auc: 0.986323, prec: 0.4819, rec: 0.9302, F1score: 0.6349, clsf_loss: 3.35910294779751e-06
Epoch 12 -------------------------------------------------------------------------
Training => auc: 0.994758, clsf_pos: 7.754294983897125e-07, clsf_neg: 1.807113569896046e-08, vat_loss: 2.980922531037322e-08
            Threshold is set to 0.30612802505493164
            Min. Probailities on test set with label 1: 0.30608057975769043
Testing ==> auc: 0.986723, prec: 0.0005, rec: 0.9767, F1score: 0.0009, clsf_loss: 3.2754285257396987e-06
Epoch 13 -------------------------------------------------------------------------
Training => auc: 0.993531, clsf_pos: 6.798949812036881e-07, clsf_neg: 1.9736088319177725e-08, vat_loss: 2.76039138213946e-08
            Threshold is set to 0.3031765818595886
            Min. Probailities on test set with label 1: 0.3022073805332184
Testing ==> auc: 0.993384, prec: 1.0000, rec: 0.9302, F1score: 0.9639, clsf_loss: 3.140434273518622e-06
Epoch 14 -------------------------------------------------------------------------
Training => auc: 0.994143, clsf_pos: 4.630934142824117e-07, clsf_neg: 2.0692370483743616e-08, vat_loss: 1.2463734933021442e-08
            Threshold is set to 0.30086997151374817
            Min. Probailities on test set with label 1: 0.3005883991718292
Testing ==> auc: 0.999456, prec: 0.6452, rec: 0.9302, F1score: 0.7619, clsf_loss: 3.1016616048873402e-06
Epoch 15 -------------------------------------------------------------------------
Training => auc: 0.993626, clsf_pos: 4.205066090889886e-07, clsf_neg: 2.2133075816554992e-08, vat_loss: 3.899674805296627e-09
            Threshold is set to 0.2989819049835205
            Min. Probailities on test set with label 1: 0.2983814775943756
Testing ==> auc: 0.999949, prec: 0.9091, rec: 0.9302, F1score: 0.9195, clsf_loss: 3.0290896120277466e-06
Epoch 16 -------------------------------------------------------------------------
Training => auc: 0.997771, clsf_pos: 3.8649605471619e-07, clsf_neg: 2.3499932666481982e-08, vat_loss: 1.8044712390974382e-09
            Threshold is set to 0.2975502014160156
            Min. Probailities on test set with label 1: 0.29637691378593445
Testing ==> auc: 0.991591, prec: 1.0000, rec: 0.9302, F1score: 0.9639, clsf_loss: 2.9834395718353335e-06
Epoch 17 -------------------------------------------------------------------------
Training => auc: 0.998737, clsf_pos: 3.495962914712436e-07, clsf_neg: 2.4890972838420566e-08, vat_loss: -2.1232832159157056e-10
            Threshold is set to 0.296190470457077
            Min. Probailities on test set with label 1: 0.2947433888912201
Testing ==> auc: 0.999931, prec: 0.9302, rec: 0.9302, F1score: 0.9302, clsf_loss: 2.9352722776820883e-06
Epoch 18 -------------------------------------------------------------------------
Training => auc: 0.990379, clsf_pos: 3.297931812085153e-07, clsf_neg: 2.6243910156154016e-08, vat_loss: -1.021291162750515e-09
            Threshold is set to 0.29332292079925537
            Min. Probailities on test set with label 1: 0.2929268479347229
Testing ==> auc: 0.999932, prec: 0.4615, rec: 0.9767, F1score: 0.6269, clsf_loss: 2.87985994873452e-06
Epoch 19 -------------------------------------------------------------------------
Training => auc: 0.991965, clsf_pos: 3.316753804938344e-07, clsf_neg: 2.7526240842234984e-08, vat_loss: -3.294947648058155e-10
            Threshold is set to 0.2920648157596588
            Min. Probailities on test set with label 1: 0.29122233390808105
Testing ==> auc: 0.999937, prec: 0.9091, rec: 0.9302, F1score: 0.9195, clsf_loss: 2.8342333280306775e-06
Epoch 20 -------------------------------------------------------------------------
Training => auc: 0.997587, clsf_pos: 2.9844633786524355e-07, clsf_neg: 2.832858569945529e-08, vat_loss: -6.441684963220951e-10
            Threshold is set to 0.2912885546684265
            Min. Probailities on test set with label 1: 0.29051080346107483
Testing ==> auc: 0.999929, prec: 1.0000, rec: 0.9302, F1score: 0.9639, clsf_loss: 2.8144836505816784e-06
Epoch 21 -------------------------------------------------------------------------
Training => auc: 0.997932, clsf_pos: 2.7652581024995015e-07, clsf_neg: 2.8767358273285026e-08, vat_loss: -3.1566851355080416e-09
            Threshold is set to 0.2902914583683014
            Min. Probailities on test set with label 1: 0.2899259924888611
Testing ==> auc: 0.999930, prec: 0.5526, rec: 0.9767, F1score: 0.7059, clsf_loss: 2.797301931423135e-06
Epoch 22 -------------------------------------------------------------------------
Training => auc: 0.996853, clsf_pos: 2.911145315920294e-07, clsf_neg: 2.8627152204308004e-08, vat_loss: -7.219442821337907e-10
            Threshold is set to 0.2893320322036743
            Min. Probailities on test set with label 1: 0.2881564199924469
Testing ==> auc: 0.999935, prec: 0.8000, rec: 0.9302, F1score: 0.8602, clsf_loss: 2.7498745112097822e-06
Epoch 23 -------------------------------------------------------------------------
Training => auc: 0.995743, clsf_pos: 2.9539253887378436e-07, clsf_neg: 2.8416110353646218e-08, vat_loss: 1.1558279888745915e-09
            Threshold is set to 0.28856712579727173
            Min. Probailities on test set with label 1: 0.28838682174682617
Testing ==> auc: 0.999936, prec: 0.5250, rec: 0.9767, F1score: 0.6829, clsf_loss: 2.755503373919055e-06
'''
