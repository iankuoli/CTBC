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



torch.cuda.set_device(0)



## Model Declaration



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
    
    
    
    
    
    
## Parameters Settings




#
# Classifier
# ---------------------
## focal loss
alpha = 1e-4
gamma_pos = 8
gamma_neg = 2
learn_rate = 1e-5

#
# VAT
# ---------------------
vat_xi = 1e-6
vat_eps = 10
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
        tmp = target.reshape(-1, 1)
        onehot_target = torch.cat([1-tmp, tmp], dim=1)
        
        #
        # Update classifier on real samples
        #
        optim_clsfr.zero_grad()
        
        vat_kld = vat_loss(classifier, data)
        
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
print('vat_eps = {}'.format(vat_eps))
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
                       'saved_models/VATConv_heter_clsfr_xi{}_eps{}_focal{}{}_BestAUC'.
                       format(int(-math.log10(vat_xi)), int(-math.log10(vat_eps)), gamma_pos, gamma_neg))
        if thres > max_thres:
            max_thres = thres if thres > max_thres else max_thres
            torch.save({'epoch': epoch,
                        'model_state_dict': classifier.state_dict(),
                        'optimizer_state_dict': optim_clsfr.state_dict(),
                        'loss': focal_loss,
                       }, 
                       'saved_models/VATConv_heter_clsfr_xi{}_eps{}_focal{}{}_BestThres'.
                       format(int(-math.log10(vat_xi)), int(-math.log10(vat_eps)), gamma_pos, gamma_neg))
        

        
        
        
        
        
        
       
    
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
vat_eps = 10
vat_ip = 1


Epoch 0 -------------------------------------------------------------------------
Training => auc: 0.574730, clsf_pos: 3.670465957839042e-06, clsf_neg: 2.535563453420764e-06, vat_loss: 2.9527848255384015e-06
            Threshold is set to 0.39246976375579834
            Min. Probailities on test set with label 1: 0.4104625880718231
Testing ==> auc: 0.545064, prec: 0.0002, rec: 1.0000, F1score: 0.0004, clsf_loss: 8.004973096831236e-06
Epoch 1 -------------------------------------------------------------------------
Training => auc: 0.554478, clsf_pos: 3.1401887099491432e-06, clsf_neg: 1.892147452053905e-06, vat_loss: 1.8608039908940555e-06
            Threshold is set to 0.3951260447502136
            Min. Probailities on test set with label 1: 0.4026031494140625
Testing ==> auc: 0.564514, prec: 0.0002, rec: 1.0000, F1score: 0.0004, clsf_loss: 7.512907359341625e-06
Epoch 2 -------------------------------------------------------------------------
Training => auc: 0.557423, clsf_pos: 2.6840248210646678e-06, clsf_neg: 1.2862835774285486e-06, vat_loss: 1.4325304391604732e-06
            Threshold is set to 0.38241174817085266
            Min. Probailities on test set with label 1: 0.38853564858436584
Testing ==> auc: 0.414092, prec: 0.0002, rec: 1.0000, F1score: 0.0004, clsf_loss: 6.731064786436036e-06
Epoch 3 -------------------------------------------------------------------------
Training => auc: 0.572484, clsf_pos: 2.3560849058412714e-06, clsf_neg: 9.312494171354047e-07, vat_loss: 8.152749728651543e-07
            Threshold is set to 0.3787906765937805
            Min. Probailities on test set with label 1: 0.3819650709629059
Testing ==> auc: 0.625451, prec: 0.0002, rec: 1.0000, F1score: 0.0004, clsf_loss: 6.369015409291023e-06
Epoch 4 -------------------------------------------------------------------------
Training => auc: 0.550143, clsf_pos: 2.1388916593423346e-06, clsf_neg: 5.431582508208521e-07, vat_loss: 5.936671527706494e-07
            Threshold is set to 0.3623935878276825
            Min. Probailities on test set with label 1: 0.36901241540908813
Testing ==> auc: 0.638675, prec: 0.0002, rec: 1.0000, F1score: 0.0004, clsf_loss: 5.724784841731889e-06
Epoch 5 -------------------------------------------------------------------------
Training => auc: 0.587381, clsf_pos: 2.180274350394029e-06, clsf_neg: 2.6945190256810747e-07, vat_loss: 3.994106236859807e-07
            Threshold is set to 0.349521279335022
            Min. Probailities on test set with label 1: 0.3533634543418884
Testing ==> auc: 0.998457, prec: 0.0002, rec: 1.0000, F1score: 0.0004, clsf_loss: 5.000239070795942e-06
Epoch 6 -------------------------------------------------------------------------
Training => auc: 0.581024, clsf_pos: 2.484975084371399e-06, clsf_neg: 6.839051991391898e-08, vat_loss: 2.3773905866164569e-07
            Threshold is set to 0.3267822563648224
            Min. Probailities on test set with label 1: 0.3285509943962097
Testing ==> auc: 0.998978, prec: 0.0002, rec: 1.0000, F1score: 0.0004, clsf_loss: 3.983134320151294e-06
Epoch 7 -------------------------------------------------------------------------
Training => auc: 0.649637, clsf_pos: 3.1815347938390914e-06, clsf_neg: 1.759528700517876e-08, vat_loss: 6.316285805496591e-08
            Threshold is set to 0.30604761838912964
            Min. Probailities on test set with label 1: 0.29599472880363464
Testing ==> auc: 0.158776, prec: 0.0000, rec: 0.2326, F1score: 0.0001, clsf_loss: 3.4083282116625924e-06
Epoch 8 -------------------------------------------------------------------------
Training => auc: 0.955170, clsf_pos: 1.9605188299465226e-06, clsf_neg: 2.1537772454394144e-08, vat_loss: 2.7926697399038858e-08
            Threshold is set to 0.2870722711086273
            Min. Probailities on test set with label 1: 0.30464884638786316
Testing ==> auc: 0.999924, prec: 0.0002, rec: 1.0000, F1score: 0.0004, clsf_loss: 3.2208247375820065e-06
Epoch 9 -------------------------------------------------------------------------
Training => auc: 0.985912, clsf_pos: 1.430887209608045e-06, clsf_neg: 2.056289538643341e-08, vat_loss: 2.0340033657362255e-08
            Threshold is set to 0.3023325502872467
            Min. Probailities on test set with label 1: 0.30224722623825073
Testing ==> auc: 0.999878, prec: 0.1654, rec: 0.9767, F1score: 0.2828, clsf_loss: 3.15160355057742e-06
Epoch 10 -------------------------------------------------------------------------
Training => auc: 0.992995, clsf_pos: 1.124245841310767e-06, clsf_neg: 2.182451552812381e-08, vat_loss: 9.981866000430273e-09
            Threshold is set to 0.3004523515701294
            Min. Probailities on test set with label 1: 0.300629198551178
Testing ==> auc: 0.999915, prec: 0.0858, rec: 1.0000, F1score: 0.1581, clsf_loss: 3.0754952149436576e-06
Epoch 11 -------------------------------------------------------------------------
Training => auc: 0.988786, clsf_pos: 1.0725121910581947e-06, clsf_neg: 2.2891796902513306e-08, vat_loss: 9.417193247429623e-09
            Threshold is set to 0.29826754331588745
            Min. Probailities on test set with label 1: 0.2977883517742157
Testing ==> auc: 0.999916, prec: 0.8163, rec: 0.9302, F1score: 0.8696, clsf_loss: 3.019358928213478e-06
Epoch 12 -------------------------------------------------------------------------
Training => auc: 0.993733, clsf_pos: 8.661172046231513e-07, clsf_neg: 2.404109622489159e-08, vat_loss: 9.644201881542358e-09
            Threshold is set to 0.29626405239105225
            Min. Probailities on test set with label 1: 0.29460155963897705
Testing ==> auc: 0.999883, prec: 0.9091, rec: 0.9302, F1score: 0.9195, clsf_loss: 2.9278678539412795e-06
Epoch 13 -------------------------------------------------------------------------
Training => auc: 0.988211, clsf_pos: 8.209780730794591e-07, clsf_neg: 2.4657641262137986e-08, vat_loss: 1.3842054169543871e-09
            Threshold is set to 0.29422399401664734
            Min. Probailities on test set with label 1: 0.29401543736457825
Testing ==> auc: 0.999933, prec: 0.4819, rec: 0.9302, F1score: 0.6349, clsf_loss: 2.9081152206344996e-06
Epoch 14 -------------------------------------------------------------------------
Training => auc: 0.994309, clsf_pos: 7.618201607328956e-07, clsf_neg: 2.560307521548566e-08, vat_loss: 7.812189117828439e-09
            Threshold is set to 0.29367294907569885
            Min. Probailities on test set with label 1: 0.2931717038154602
Testing ==> auc: 0.999954, prec: 0.7692, rec: 0.9302, F1score: 0.8421, clsf_loss: 2.885116145989741e-06
Epoch 15 -------------------------------------------------------------------------
Training => auc: 0.988900, clsf_pos: 7.554402259302151e-07, clsf_neg: 2.6199396430115485e-08, vat_loss: -2.9667721057613505e-10
            Threshold is set to 0.29194456338882446
            Min. Probailities on test set with label 1: 0.29194220900535583
Testing ==> auc: 0.999964, prec: 0.1342, rec: 0.9767, F1score: 0.2360, clsf_loss: 2.847768882929813e-06
Epoch 16 -------------------------------------------------------------------------
Training => auc: 0.997230, clsf_pos: 7.325620572373737e-07, clsf_neg: 2.6448958578839665e-08, vat_loss: -5.489921850454493e-10
            Threshold is set to 0.29123005270957947
            Min. Probailities on test set with label 1: 0.2906038165092468
Testing ==> auc: 0.999969, prec: 1.0000, rec: 0.9302, F1score: 0.9639, clsf_loss: 2.813964783854317e-06
Epoch 17 -------------------------------------------------------------------------
Training => auc: 0.987397, clsf_pos: 6.651636113019777e-07, clsf_neg: 2.5457637775616604e-08, vat_loss: -2.1534543037660114e-09
            Threshold is set to 0.28966960310935974
            Min. Probailities on test set with label 1: 0.2900729477405548
Testing ==> auc: 0.999976, prec: 0.0002, rec: 1.0000, F1score: 0.0004, clsf_loss: 2.7975831926596584e-06
Epoch 18 -------------------------------------------------------------------------
Training => auc: 0.998942, clsf_pos: 6.101051326368179e-07, clsf_neg: 2.4231090023363322e-08, vat_loss: -9.888927454682062e-10
            Threshold is set to 0.28989431262016296
            Min. Probailities on test set with label 1: 0.2895880937576294
Testing ==> auc: 0.999971, prec: 0.7547, rec: 0.9302, F1score: 0.8333, clsf_loss: 2.786780441965675e-06
Epoch 19 -------------------------------------------------------------------------
Training => auc: 0.993912, clsf_pos: 5.934988962508214e-07, clsf_neg: 2.4472967652400257e-08, vat_loss: 8.060470624293714e-10
            Threshold is set to 0.28894200921058655
            Min. Probailities on test set with label 1: 0.28926926851272583
Testing ==> auc: 0.999967, prec: 0.0002, rec: 1.0000, F1score: 0.0004, clsf_loss: 2.774818085526931e-06
Epoch 20 -------------------------------------------------------------------------
Training => auc: 0.989292, clsf_pos: 6.125061418060795e-07, clsf_neg: 2.5388771973666735e-08, vat_loss: 3.3318372505419802e-09
            Threshold is set to 0.2885265052318573
            Min. Probailities on test set with label 1: 0.2884664535522461
Testing ==> auc: 0.999977, prec: 0.2530, rec: 0.9767, F1score: 0.4019, clsf_loss: 2.7476903596834745e-06
Epoch 21 -------------------------------------------------------------------------
Training => auc: 0.993213, clsf_pos: 6.675595614069607e-07, clsf_neg: 2.6614731751806175e-08, vat_loss: 2.26859930840817e-09
            Threshold is set to 0.28821632266044617
            Min. Probailities on test set with label 1: 0.2885933220386505
Testing ==> auc: 0.999980, prec: 0.0325, rec: 1.0000, F1score: 0.0630, clsf_loss: 2.7510125164553756e-06
Epoch 22 -------------------------------------------------------------------------
Training => auc: 0.994295, clsf_pos: 5.822145112688304e-07, clsf_neg: 2.356888728627382e-08, vat_loss: 7.255721357068978e-09
            Threshold is set to 0.28800612688064575
            Min. Probailities on test set with label 1: 0.2882978916168213
Testing ==> auc: 0.999981, prec: 0.0347, rec: 1.0000, F1score: 0.0671, clsf_loss: 2.7426167434896342e-06
Epoch 23 -------------------------------------------------------------------------
Training => auc: 0.994975, clsf_pos: 6.267316621233476e-07, clsf_neg: 2.6333658809107874e-08, vat_loss: 7.746707719746837e-10
            Threshold is set to 0.2877531945705414
            Min. Probailities on test set with label 1: 0.2877027094364166
Testing ==> auc: 0.999978, prec: 0.3500, rec: 0.9767, F1score: 0.5153, clsf_loss: 2.7350247364665847e-06
Epoch 24 -------------------------------------------------------------------------
Training => auc: 0.997221, clsf_pos: 6.477922056546959e-07, clsf_neg: 2.5384409241269168e-08, vat_loss: -1.1706510205655718e-09
            Threshold is set to 0.2875394821166992
            Min. Probailities on test set with label 1: 0.2875070571899414
Testing ==> auc: 0.999984, prec: 0.3307, rec: 0.9767, F1score: 0.4941, clsf_loss: 2.7259081889496883e-06
Epoch 25 -------------------------------------------------------------------------
Training => auc: 0.998806, clsf_pos: 4.79180471302243e-07, clsf_neg: 1.9301463538567987e-08, vat_loss: 1.7254519590892414e-08
            Threshold is set to 0.2879907190799713
            Min. Probailities on test set with label 1: 0.28771093487739563
Testing ==> auc: 0.999966, prec: 0.7018, rec: 0.9302, F1score: 0.8000, clsf_loss: 2.731654603849165e-06
Epoch 26 -------------------------------------------------------------------------
Training => auc: 0.994146, clsf_pos: 4.993003699382825e-07, clsf_neg: 2.1148585105379425e-08, vat_loss: 1.0085705604012674e-08
            Threshold is set to 0.28753945231437683
            Min. Probailities on test set with label 1: 0.28775325417518616
Testing ==> auc: 0.999984, prec: 0.0475, rec: 1.0000, F1score: 0.0906, clsf_loss: 2.7274252261122456e-06
Epoch 27 -------------------------------------------------------------------------
Training => auc: 0.999500, clsf_pos: 5.078230174149212e-07, clsf_neg: 2.23540013166712e-08, vat_loss: 4.754411975227413e-09
            Threshold is set to 0.2876105010509491
            Min. Probailities on test set with label 1: 0.2873661518096924
Testing ==> auc: 0.999971, prec: 0.6557, rec: 0.9302, F1score: 0.7692, clsf_loss: 2.723696070461301e-06
Epoch 28 -------------------------------------------------------------------------
Training => auc: 0.999458, clsf_pos: 5.040044470661087e-07, clsf_neg: 2.2658875664660627e-08, vat_loss: 4.1618419821531916e-09
            Threshold is set to 0.2873993515968323
            Min. Probailities on test set with label 1: 0.28816208243370056
Testing ==> auc: 0.999981, prec: 0.0002, rec: 1.0000, F1score: 0.0004, clsf_loss: 2.735763018790749e-06
Epoch 29 -------------------------------------------------------------------------
Training => auc: 0.996464, clsf_pos: 4.7004411385387357e-07, clsf_neg: 2.1200705191404268e-08, vat_loss: 1.1525554732827459e-08
            Threshold is set to 0.28722161054611206
            Min. Probailities on test set with label 1: 0.28721338510513306
Testing ==> auc: 0.999969, prec: 0.2917, rec: 0.9767, F1score: 0.4492, clsf_loss: 2.723712896113284e-06
Epoch 30 -------------------------------------------------------------------------
Training => auc: 0.996583, clsf_pos: 5.480039249050606e-07, clsf_neg: 2.3014647965169388e-08, vat_loss: 4.5689367844659046e-09
            Threshold is set to 0.2871541380882263
            Min. Probailities on test set with label 1: 0.2873028814792633
Testing ==> auc: 0.999981, prec: 0.0447, rec: 1.0000, F1score: 0.0857, clsf_loss: 2.718421001191018e-06
Epoch 31 -------------------------------------------------------------------------
Training => auc: 0.992458, clsf_pos: 6.09143739893625e-07, clsf_neg: 2.5655078061959102e-08, vat_loss: 1.8348257635469878e-10
            Threshold is set to 0.28669673204421997
            Min. Probailities on test set with label 1: 0.2872653305530548
Testing ==> auc: 0.999981, prec: 0.0002, rec: 1.0000, F1score: 0.0004, clsf_loss: 2.717001962082577e-06
Epoch 32 -------------------------------------------------------------------------
Training => auc: 0.993473, clsf_pos: 6.180963509905268e-07, clsf_neg: 2.467510462622613e-08, vat_loss: 1.8707273508056232e-09
            Threshold is set to 0.28659549355506897
            Min. Probailities on test set with label 1: 0.2872253358364105
Testing ==> auc: 0.999981, prec: 0.0182, rec: 1.0000, F1score: 0.0357, clsf_loss: 2.708181227717432e-06
Epoch 33 -------------------------------------------------------------------------
Training => auc: 0.993727, clsf_pos: 6.464184707510867e-07, clsf_neg: 2.652299002647851e-08, vat_loss: 2.0976893111424033e-08
            Threshold is set to 0.28639325499534607
            Min. Probailities on test set with label 1: 0.2863723635673523
Testing ==> auc: 0.999969, prec: 0.2456, rec: 0.9767, F1score: 0.3925, clsf_loss: 2.701232688195887e-06
Epoch 34 -------------------------------------------------------------------------
Training => auc: 0.994258, clsf_pos: 6.310406774900912e-07, clsf_neg: 2.5019904370537915e-08, vat_loss: 3.2534006599860277e-09
            Threshold is set to 0.2862880527973175
            Min. Probailities on test set with label 1: 0.28660327196121216
Testing ==> auc: 0.999965, prec: 0.0002, rec: 1.0000, F1score: 0.0004, clsf_loss: 2.707503426790936e-06
Epoch 35 -------------------------------------------------------------------------
Training => auc: 0.996413, clsf_pos: 6.713984248563065e-07, clsf_neg: 2.5384384372273416e-08, vat_loss: 1.4056165120734931e-08
            Threshold is set to 0.2862895727157593
            Min. Probailities on test set with label 1: 0.286332905292511
Testing ==> auc: 0.999956, prec: 0.0543, rec: 1.0000, F1score: 0.1030, clsf_loss: 2.700244749576086e-06
Epoch 36 -------------------------------------------------------------------------
Training => auc: 0.993009, clsf_pos: 6.879954526084475e-07, clsf_neg: 2.4626604755439985e-08, vat_loss: 1.7397070450897445e-09
            Threshold is set to 0.28618279099464417
            Min. Probailities on test set with label 1: 0.2862786650657654
Testing ==> auc: 0.999972, prec: 0.0460, rec: 1.0000, F1score: 0.0879, clsf_loss: 2.695080411285744e-06
Epoch 37 -------------------------------------------------------------------------
Training => auc: 0.993991, clsf_pos: 6.253453648241702e-07, clsf_neg: 2.278219390916547e-08, vat_loss: 6.026174226292369e-09
            Threshold is set to 0.28600025177001953
            Min. Probailities on test set with label 1: 0.28649571537971497
Testing ==> auc: 0.999969, prec: 0.0002, rec: 1.0000, F1score: 0.0004, clsf_loss: 2.700676077438402e-06
Epoch 38 -------------------------------------------------------------------------
Training => auc: 0.992755, clsf_pos: 8.240361921707517e-07, clsf_neg: 2.651173680590091e-08, vat_loss: 3.341751764196488e-08
            Threshold is set to 0.2860001027584076
            Min. Probailities on test set with label 1: 0.28659942746162415
Testing ==> auc: 0.999975, prec: 0.0002, rec: 1.0000, F1score: 0.0004, clsf_loss: 2.6996497126674512e-06
Epoch 39 -------------------------------------------------------------------------
Training => auc: 0.998120, clsf_pos: 6.816115387664468e-07, clsf_neg: 2.4349176896976132e-08, vat_loss: 1.4993639663174463e-09
            Threshold is set to 0.28606733679771423
            Min. Probailities on test set with label 1: 0.2860078513622284
Testing ==> auc: 0.999966, prec: 0.2958, rec: 0.9767, F1score: 0.4541, clsf_loss: 2.692977659535245e-06
Epoch 40 -------------------------------------------------------------------------
Training => auc: 0.995093, clsf_pos: 7.211442607513163e-07, clsf_neg: 2.3616678390681045e-08, vat_loss: 4.0578196358609375e-10
            Threshold is set to 0.2858133018016815
            Min. Probailities on test set with label 1: 0.28610289096832275
Testing ==> auc: 0.999976, prec: 0.0002, rec: 1.0000, F1score: 0.0004, clsf_loss: 2.6890575099969283e-06
Epoch 41 -------------------------------------------------------------------------
Training => auc: 0.996530, clsf_pos: 6.664403144895914e-07, clsf_neg: 2.1520525805840407e-08, vat_loss: 2.8233366755614497e-09
            Threshold is set to 0.2857249677181244
            Min. Probailities on test set with label 1: 0.28595802187919617
Testing ==> auc: 0.999972, prec: 0.0002, rec: 1.0000, F1score: 0.0004, clsf_loss: 2.68774260803184e-06
Epoch 42 -------------------------------------------------------------------------
Training => auc: 0.998512, clsf_pos: 6.178906915010884e-07, clsf_neg: 2.3387512371186858e-08, vat_loss: 8.993468192386445e-09
            Threshold is set to 0.2858576774597168
            Min. Probailities on test set with label 1: 0.2858370840549469
Testing ==> auc: 0.999965, prec: 0.1963, rec: 0.9767, F1score: 0.3268, clsf_loss: 2.683783350221347e-06
Epoch 43 -------------------------------------------------------------------------
Training => auc: 0.991190, clsf_pos: 6.394775482476689e-07, clsf_neg: 2.2482058881223566e-08, vat_loss: 7.032329385481262e-09
            Threshold is set to 0.28518858551979065
            Min. Probailities on test set with label 1: 0.2854616641998291
Testing ==> auc: 0.999969, prec: 0.0002, rec: 1.0000, F1score: 0.0004, clsf_loss: 2.6729856017482234e-06
Epoch 44 -------------------------------------------------------------------------
Training => auc: 0.996856, clsf_pos: 5.871658572687011e-07, clsf_neg: 2.298934376199213e-08, vat_loss: 1.9398831430095242e-09
            Threshold is set to 0.28558310866355896
            Min. Probailities on test set with label 1: 0.2859695255756378
Testing ==> auc: 0.999970, prec: 0.0002, rec: 1.0000, F1score: 0.0004, clsf_loss: 2.6814955162990373e-06
Epoch 45 -------------------------------------------------------------------------
Training => auc: 0.999586, clsf_pos: 5.826896654070879e-07, clsf_neg: 2.2490807438657612e-08, vat_loss: 4.22229806673613e-09
            Threshold is set to 0.2857277989387512
            Min. Probailities on test set with label 1: 0.28660258650779724
Testing ==> auc: 0.999966, prec: 0.0002, rec: 1.0000, F1score: 0.0004, clsf_loss: 2.691471081561758e-06
Epoch 46 -------------------------------------------------------------------------
Training => auc: 0.994644, clsf_pos: 5.635105253531947e-07, clsf_neg: 2.306896185189089e-08, vat_loss: 6.34983976510739e-09
            Threshold is set to 0.28547361493110657
            Min. Probailities on test set with label 1: 0.2856074273586273
Testing ==> auc: 0.999964, prec: 0.0109, rec: 1.0000, F1score: 0.0216, clsf_loss: 2.6797770260600373e-06
Epoch 47 -------------------------------------------------------------------------
Training => auc: 0.997902, clsf_pos: 5.495411983247322e-07, clsf_neg: 2.1639175784571307e-08, vat_loss: 5.956710680266042e-09
            Threshold is set to 0.2855682373046875
            Min. Probailities on test set with label 1: 0.28553593158721924
Testing ==> auc: 0.999957, prec: 0.2121, rec: 0.9767, F1score: 0.3485, clsf_loss: 2.678234750419506e-06
Epoch 48 -------------------------------------------------------------------------
Training => auc: 0.995761, clsf_pos: 6.137742047940264e-07, clsf_neg: 2.3626562040135468e-08, vat_loss: -5.954557957821294e-10
            Threshold is set to 0.28542983531951904
            Min. Probailities on test set with label 1: 0.2854917049407959
Testing ==> auc: 0.999958, prec: 0.0468, rec: 1.0000, F1score: 0.0894, clsf_loss: 2.6753762085718336e-06
Epoch 49 -------------------------------------------------------------------------
Training => auc: 0.997419, clsf_pos: 5.978725425848097e-07, clsf_neg: 2.162660805993255e-08, vat_loss: 7.45203287966234e-10
            Threshold is set to 0.28534770011901855
            Min. Probailities on test set with label 1: 0.2854498326778412
Testing ==> auc: 0.999961, prec: 0.0002, rec: 1.0000, F1score: 0.0004, clsf_loss: 2.6758850708574755e-06
Epoch 50 -------------------------------------------------------------------------
Training => auc: 0.995259, clsf_pos: 5.83532539621956e-07, clsf_neg: 2.2419094136694184e-08, vat_loss: 1.1573851876889307e-09
            Threshold is set to 0.28516316413879395
            Min. Probailities on test set with label 1: 0.28533172607421875
Testing ==> auc: 0.999946, prec: 0.0002, rec: 1.0000, F1score: 0.0004, clsf_loss: 2.6725890620582504e-06
Epoch 51 -------------------------------------------------------------------------
Training => auc: 0.999055, clsf_pos: 5.951199568698939e-07, clsf_neg: 2.2241479769036232e-08, vat_loss: 2.764316830994318e-10
            Threshold is set to 0.2852470278739929
            Min. Probailities on test set with label 1: 0.2854788303375244
Testing ==> auc: 0.999961, prec: 0.0327, rec: 1.0000, F1score: 0.0633, clsf_loss: 2.6703175990405725e-06
Epoch 52 -------------------------------------------------------------------------
Training => auc: 0.999984, clsf_pos: 5.963340754533419e-07, clsf_neg: 2.1991281684563546e-08, vat_loss: 1.0807852390826156e-09
            Threshold is set to 0.285605788230896
            Min. Probailities on test set with label 1: 0.2852018177509308
Testing ==> auc: 0.999971, prec: 0.4719, rec: 0.9767, F1score: 0.6364, clsf_loss: 2.668910155989579e-06
Epoch 53 -------------------------------------------------------------------------
Training => auc: 0.997323, clsf_pos: 6.589112899746397e-07, clsf_neg: 2.1329254806801146e-08, vat_loss: -1.5433070377213198e-09
            Threshold is set to 0.2849288880825043
            Min. Probailities on test set with label 1: 0.2853204309940338
Testing ==> auc: 0.999977, prec: 0.0347, rec: 1.0000, F1score: 0.0671, clsf_loss: 2.6607390282151755e-06
Epoch 54 -------------------------------------------------------------------------
Training => auc: 0.995942, clsf_pos: 5.582677431448246e-07, clsf_neg: 1.9667911743681543e-08, vat_loss: 2.816535671357201e-09
            Threshold is set to 0.2848183214664459
            Min. Probailities on test set with label 1: 0.28502506017684937
Testing ==> auc: 0.999969, prec: 0.0002, rec: 1.0000, F1score: 0.0004, clsf_loss: 2.6626705675880658e-06
Epoch 55 -------------------------------------------------------------------------
Training => auc: 0.999996, clsf_pos: 4.85752934764605e-07, clsf_neg: 1.91000744109715e-08, vat_loss: 5.009484826956623e-09
            Threshold is set to 0.28556254506111145
            Min. Probailities on test set with label 1: 0.284924179315567
Testing ==> auc: 0.999950, prec: 0.8333, rec: 0.9302, F1score: 0.8791, clsf_loss: 2.6633115339791402e-06
Epoch 56 -------------------------------------------------------------------------
Training => auc: 0.990730, clsf_pos: 5.796858886242262e-07, clsf_neg: 2.2292832468906454e-08, vat_loss: -7.257191514398187e-10
            Threshold is set to 0.28456512093544006
            Min. Probailities on test set with label 1: 0.2849639356136322
Testing ==> auc: 0.999961, prec: 0.0002, rec: 1.0000, F1score: 0.0004, clsf_loss: 2.6582761165627744e-06
Epoch 57 -------------------------------------------------------------------------
Training => auc: 0.998674, clsf_pos: 5.871878556718002e-07, clsf_neg: 2.05727008761869e-08, vat_loss: -1.8129039935921298e-10
            Threshold is set to 0.28483447432518005
            Min. Probailities on test set with label 1: 0.28493642807006836
Testing ==> auc: 0.999950, prec: 0.0363, rec: 1.0000, F1score: 0.0701, clsf_loss: 2.6579702989693033e-06
Epoch 58 -------------------------------------------------------------------------
Training => auc: 0.995456, clsf_pos: 5.753671530328575e-07, clsf_neg: 2.088242467834789e-08, vat_loss: -1.71419892169844e-10
            Threshold is set to 0.28455784916877747
            Min. Probailities on test set with label 1: 0.28473639488220215
Testing ==> auc: 0.999956, prec: 0.0352, rec: 1.0000, F1score: 0.0681, clsf_loss: 2.6507470920478227e-06
Epoch 59 -------------------------------------------------------------------------
Training => auc: 0.997256, clsf_pos: 4.771672479364497e-07, clsf_neg: 1.8220809749891487e-08, vat_loss: 6.958218001784644e-09
            Threshold is set to 0.2846778333187103
            Min. Probailities on test set with label 1: 0.28460901975631714
Testing ==> auc: 0.999943, prec: 0.3853, rec: 0.9767, F1score: 0.5526, clsf_loss: 2.6537593385000946e-06
Epoch 60 -------------------------------------------------------------------------
Training => auc: 0.995489, clsf_pos: 5.483572635966993e-07, clsf_neg: 1.9766615011462818e-08, vat_loss: 1.9863177769252616e-09
            Threshold is set to 0.28458118438720703
            Min. Probailities on test set with label 1: 0.28521648049354553
Testing ==> auc: 0.999956, prec: 0.0002, rec: 1.0000, F1score: 0.0004, clsf_loss: 2.6578002234600717e-06
Epoch 61 -------------------------------------------------------------------------
Training => auc: 0.999850, clsf_pos: 4.319331310398411e-07, clsf_neg: 2.2256404719200873e-08, vat_loss: 4.177286072604147e-09
            Threshold is set to 0.28497204184532166
            Min. Probailities on test set with label 1: 0.28482866287231445
Testing ==> auc: 0.999935, prec: 0.5316, rec: 0.9767, F1score: 0.6885, clsf_loss: 2.659095116541721e-06
Epoch 62 -------------------------------------------------------------------------
Training => auc: 0.996194, clsf_pos: 5.521581556422461e-07, clsf_neg: 2.1300911257071675e-08, vat_loss: -1.506422486752257e-10
            Threshold is set to 0.28461626172065735
            Min. Probailities on test set with label 1: 0.2847774922847748
Testing ==> auc: 0.999953, prec: 0.0334, rec: 1.0000, F1score: 0.0647, clsf_loss: 2.6550180791673483e-06
Epoch 63 -------------------------------------------------------------------------
Training => auc: 0.996893, clsf_pos: 5.693599973710661e-07, clsf_neg: 2.1160134977549205e-08, vat_loss: 2.4560697919895347e-10
            Threshold is set to 0.28449371457099915
            Min. Probailities on test set with label 1: 0.284782350063324
Testing ==> auc: 0.999948, prec: 0.0306, rec: 1.0000, F1score: 0.0595, clsf_loss: 2.647928340593353e-06
Epoch 64 -------------------------------------------------------------------------
Training => auc: 0.999904, clsf_pos: 4.941612701259146e-07, clsf_neg: 1.9523730188097943e-08, vat_loss: 5.402811975585564e-09
            Threshold is set to 0.2848452925682068
            Min. Probailities on test set with label 1: 0.28482910990715027
Testing ==> auc: 0.999957, prec: 0.1027, rec: 0.9767, F1score: 0.1858, clsf_loss: 2.6560712740320014e-06
Epoch 65 -------------------------------------------------------------------------
Training => auc: 0.997777, clsf_pos: 4.0969902670440206e-07, clsf_neg: 1.855297426800462e-08, vat_loss: 1.180413988777218e-08
            Threshold is set to 0.28467097878456116
            Min. Probailities on test set with label 1: 0.2851104736328125
Testing ==> auc: 0.999964, prec: 0.0002, rec: 1.0000, F1score: 0.0004, clsf_loss: 2.65856624537264e-06
Epoch 66 -------------------------------------------------------------------------
Training => auc: 0.995338, clsf_pos: 3.930757657144568e-07, clsf_neg: 1.7961298226509825e-08, vat_loss: 1.9261031880546398e-08
            Threshold is set to 0.2845928966999054
            Min. Probailities on test set with label 1: 0.2848506569862366
Testing ==> auc: 0.999927, prec: 0.0002, rec: 1.0000, F1score: 0.0004, clsf_loss: 2.6568670818960527e-06
Epoch 67 -------------------------------------------------------------------------
Training => auc: 0.996382, clsf_pos: 4.5478992660719086e-07, clsf_neg: 2.282594380176306e-08, vat_loss: 3.21317239482255e-09
            Threshold is set to 0.2845833897590637
            Min. Probailities on test set with label 1: 0.2846226692199707
Testing ==> auc: 0.999948, prec: 0.0374, rec: 1.0000, F1score: 0.0720, clsf_loss: 2.6533380150794983e-06
Epoch 68 -------------------------------------------------------------------------
Training => auc: 0.999549, clsf_pos: 5.033666639064904e-07, clsf_neg: 2.1207059219818802e-08, vat_loss: 1.0195479877017632e-10
            Threshold is set to 0.2847689390182495
            Min. Probailities on test set with label 1: 0.28524988889694214
Testing ==> auc: 0.999953, prec: 0.0332, rec: 1.0000, F1score: 0.0642, clsf_loss: 2.652930106705753e-06
Epoch 69 -------------------------------------------------------------------------
Training => auc: 0.999953, clsf_pos: 4.804011837222788e-07, clsf_neg: 2.1006419714808544e-08, vat_loss: 4.020051846964634e-09
            Threshold is set to 0.2848860025405884
            Min. Probailities on test set with label 1: 0.2842182219028473
Testing ==> auc: 0.999930, prec: 0.9756, rec: 0.9302, F1score: 0.9524, clsf_loss: 2.642610752445762e-06
Epoch 70 -------------------------------------------------------------------------
Training => auc: 0.999005, clsf_pos: 5.645732130687975e-07, clsf_neg: 2.2598344529001224e-08, vat_loss: -7.825775028003079e-10
            Threshold is set to 0.28446337580680847
            Min. Probailities on test set with label 1: 0.28442150354385376
Testing ==> auc: 0.999949, prec: 0.1469, rec: 0.9767, F1score: 0.2553, clsf_loss: 2.645324457262177e-06
Epoch 71 -------------------------------------------------------------------------
Training => auc: 0.998820, clsf_pos: 5.549904358304047e-07, clsf_neg: 2.1186371768067147e-08, vat_loss: 9.122021193963548e-11
            Threshold is set to 0.2844851613044739
            Min. Probailities on test set with label 1: 0.28449633717536926
Testing ==> auc: 0.999929, prec: 0.0493, rec: 1.0000, F1score: 0.0939, clsf_loss: 2.649466750881402e-06
Epoch 72 -------------------------------------------------------------------------
Training => auc: 0.994613, clsf_pos: 5.622180765385565e-07, clsf_neg: 2.055937109446404e-08, vat_loss: 1.8267276580274938e-10
            Threshold is set to 0.284146249294281
            Min. Probailities on test set with label 1: 0.2844950556755066
Testing ==> auc: 0.999931, prec: 0.0002, rec: 1.0000, F1score: 0.0004, clsf_loss: 2.64714708464453e-06
Epoch 73 -------------------------------------------------------------------------
Training => auc: 0.999409, clsf_pos: 5.294569973557373e-07, clsf_neg: 2.0396134559064194e-08, vat_loss: 1.2708125662896919e-09
            Threshold is set to 0.28435665369033813
            Min. Probailities on test set with label 1: 0.2841528654098511
Testing ==> auc: 0.999937, prec: 0.2515, rec: 0.9767, F1score: 0.4000, clsf_loss: 2.6339141641074093e-06
Epoch 74 -------------------------------------------------------------------------
Training => auc: 0.998933, clsf_pos: 5.183698021937744e-07, clsf_neg: 2.1210372125324284e-08, vat_loss: 1.5587624524471266e-09
            Threshold is set to 0.2843060791492462
            Min. Probailities on test set with label 1: 0.2843382954597473
Testing ==> auc: 0.999938, prec: 0.0460, rec: 1.0000, F1score: 0.0879, clsf_loss: 2.6409636575408513e-06
Epoch 75 -------------------------------------------------------------------------
Training => auc: 0.999831, clsf_pos: 5.074413138572709e-07, clsf_neg: 1.9750567403775676e-08, vat_loss: 1.711483954558446e-09
            Threshold is set to 0.28442373871803284
            Min. Probailities on test set with label 1: 0.2841458320617676
Testing ==> auc: 0.999926, prec: 0.4828, rec: 0.9767, F1score: 0.6462, clsf_loss: 2.64096820501436e-06
Epoch 76 -------------------------------------------------------------------------
Training => auc: 0.995004, clsf_pos: 5.231132718108711e-07, clsf_neg: 2.1540635941619257e-08, vat_loss: 4.1751044288496075e-10
            Threshold is set to 0.28400948643684387
            Min. Probailities on test set with label 1: 0.28485849499702454
Testing ==> auc: 0.986353, prec: 0.0002, rec: 1.0000, F1score: 0.0004, clsf_loss: 2.660894779182854e-06
Epoch 77 -------------------------------------------------------------------------
Training => auc: 0.999954, clsf_pos: 5.356322390071e-07, clsf_neg: 2.0532695543806767e-08, vat_loss: 7.900227694257467e-10
            Threshold is set to 0.28440308570861816
            Min. Probailities on test set with label 1: 0.2840460538864136
Testing ==> auc: 0.999926, prec: 0.4200, rec: 0.9767, F1score: 0.5874, clsf_loss: 2.6359637104178546e-06
Epoch 78 -------------------------------------------------------------------------
Training => auc: 0.999964, clsf_pos: 4.966663595951104e-07, clsf_neg: 2.0232617359283722e-08, vat_loss: 1.5564188826644454e-09
            Threshold is set to 0.28456610441207886
            Min. Probailities on test set with label 1: 0.2840270400047302
Testing ==> auc: 0.999922, prec: 0.6949, rec: 0.9535, F1score: 0.8039, clsf_loss: 2.6376928872196004e-06
Epoch 79 -------------------------------------------------------------------------
Training => auc: 0.996053, clsf_pos: 5.795787387796736e-07, clsf_neg: 2.0180268123226597e-08, vat_loss: 1.1272360128655734e-10
            Threshold is set to 0.28409042954444885
            Min. Probailities on test set with label 1: 0.28411865234375
Testing ==> auc: 0.999924, prec: 0.0421, rec: 1.0000, F1score: 0.0808, clsf_loss: 2.6379375412943773e-06
Epoch 80 -------------------------------------------------------------------------
Training => auc: 0.999878, clsf_pos: 5.550396622311382e-07, clsf_neg: 2.1002207972742326e-08, vat_loss: -4.599995731546791e-11
            Threshold is set to 0.2841758728027344
            Min. Probailities on test set with label 1: 0.2840046286582947
Testing ==> auc: 0.999926, prec: 0.0923, rec: 0.9767, F1score: 0.1687, clsf_loss: 2.633559006426367e-06
Epoch 81 -------------------------------------------------------------------------
Training => auc: 0.994338, clsf_pos: 5.152453468326712e-07, clsf_neg: 2.0497893160609237e-08, vat_loss: 1.2229766088722727e-09
            Threshold is set to 0.28381937742233276
            Min. Probailities on test set with label 1: 0.2842028737068176
Testing ==> auc: 0.999928, prec: 0.0002, rec: 1.0000, F1score: 0.0004, clsf_loss: 2.638408886923571e-06
Epoch 82 -------------------------------------------------------------------------
Training => auc: 0.998548, clsf_pos: 5.136292884344584e-07, clsf_neg: 1.964040841073711e-08, vat_loss: 6.81719403150538e-10
            Threshold is set to 0.284021258354187
            Min. Probailities on test set with label 1: 0.284016489982605
Testing ==> auc: 0.999930, prec: 0.0557, rec: 0.9767, F1score: 0.1054, clsf_loss: 2.6365673875261564e-06
Epoch 83 -------------------------------------------------------------------------
Training => auc: 0.999992, clsf_pos: 5.233981710262015e-07, clsf_neg: 2.04371080059218e-08, vat_loss: 6.547440367654644e-10
            Threshold is set to 0.28432273864746094
            Min. Probailities on test set with label 1: 0.28390905261039734
Testing ==> auc: 0.999924, prec: 0.4242, rec: 0.9767, F1score: 0.5915, clsf_loss: 2.633687245179317e-06
Epoch 84 -------------------------------------------------------------------------
Training => auc: 0.999911, clsf_pos: 5.463014645101794e-07, clsf_neg: 1.9185653954423287e-08, vat_loss: 5.402213121286081e-10
            Threshold is set to 0.28412461280822754
            Min. Probailities on test set with label 1: 0.2839832305908203
Testing ==> auc: 0.999924, prec: 0.0873, rec: 0.9767, F1score: 0.1603, clsf_loss: 2.6348900519224117e-06
Epoch 85 -------------------------------------------------------------------------
Training => auc: 0.999989, clsf_pos: 5.332745445230103e-07, clsf_neg: 1.9168039600003794e-08, vat_loss: 5.954034487665183e-10
            Threshold is set to 0.2840999364852905
            Min. Probailities on test set with label 1: 0.283883273601532
Testing ==> auc: 0.999926, prec: 0.1395, rec: 0.9767, F1score: 0.2442, clsf_loss: 2.6314944534533424e-06
Epoch 86 -------------------------------------------------------------------------
Training => auc: 0.998512, clsf_pos: 4.775702677761728e-07, clsf_neg: 1.8624964681634992e-08, vat_loss: 2.421198797009083e-09
            Threshold is set to 0.28390535712242126
            Min. Probailities on test set with label 1: 0.28397586941719055
Testing ==> auc: 0.999927, prec: 0.0343, rec: 1.0000, F1score: 0.0663, clsf_loss: 2.633418489494943e-06
Epoch 87 -------------------------------------------------------------------------
Training => auc: 0.999759, clsf_pos: 4.857848807660048e-07, clsf_neg: 1.8913732802161576e-08, vat_loss: 9.222960728472174e-10
            Threshold is set to 0.2840164005756378
            Min. Probailities on test set with label 1: 0.28385889530181885
Testing ==> auc: 0.999924, prec: 0.0857, rec: 0.9767, F1score: 0.1576, clsf_loss: 2.630041990414611e-06
Epoch 88 -------------------------------------------------------------------------
Training => auc: 0.996742, clsf_pos: 4.406870743878244e-07, clsf_neg: 1.8525136979974377e-08, vat_loss: 3.765565192992426e-09
            Threshold is set to 0.2838185429573059
            Min. Probailities on test set with label 1: 0.2839060127735138
Testing ==> auc: 0.999929, prec: 0.0111, rec: 1.0000, F1score: 0.0220, clsf_loss: 2.6331883873353945e-06
Epoch 89 -------------------------------------------------------------------------
Training => auc: 0.999980, clsf_pos: 4.272574187780265e-07, clsf_neg: 2.0850210447065365e-08, vat_loss: 1.5554452170718491e-09
            Threshold is set to 0.2842084765434265
            Min. Probailities on test set with label 1: 0.2838134467601776
Testing ==> auc: 0.999925, prec: 0.5060, rec: 0.9767, F1score: 0.6667, clsf_loss: 2.6301941034034826e-06
Epoch 90 -------------------------------------------------------------------------
Training => auc: 0.997907, clsf_pos: 4.826057988793764e-07, clsf_neg: 1.808156291360774e-08, vat_loss: 2.175985835961569e-09
            Threshold is set to 0.28384336829185486
            Min. Probailities on test set with label 1: 0.2838227152824402
Testing ==> auc: 0.999927, prec: 0.0593, rec: 0.9767, F1score: 0.1119, clsf_loss: 2.6300258468836546e-06
Epoch 91 -------------------------------------------------------------------------
Training => auc: 0.993799, clsf_pos: 4.852956294598698e-07, clsf_neg: 1.9556543051635344e-08, vat_loss: 6.797054030727168e-10
            Threshold is set to 0.28354039788246155
            Min. Probailities on test set with label 1: 0.28375017642974854
Testing ==> auc: 0.999929, prec: 0.0002, rec: 1.0000, F1score: 0.0004, clsf_loss: 2.6275376967532793e-06
Epoch 92 -------------------------------------------------------------------------
Training => auc: 0.999609, clsf_pos: 4.976387799615622e-07, clsf_neg: 2.005983468222894e-08, vat_loss: 1.8974014026618136e-10
            Threshold is set to 0.2838705778121948
            Min. Probailities on test set with label 1: 0.2839057445526123
Testing ==> auc: 0.999925, prec: 0.0393, rec: 1.0000, F1score: 0.0757, clsf_loss: 2.6322522899135947e-06
Epoch 93 -------------------------------------------------------------------------
Training => auc: 0.997141, clsf_pos: 5.492021273312275e-07, clsf_neg: 1.9226499503588457e-08, vat_loss: -1.095171175968801e-09
            Threshold is set to 0.2835865020751953
            Min. Probailities on test set with label 1: 0.28354549407958984
Testing ==> auc: 0.999916, prec: 0.0830, rec: 0.9767, F1score: 0.1530, clsf_loss: 2.626227797009051e-06
Epoch 94 -------------------------------------------------------------------------
Training => auc: 0.999997, clsf_pos: 5.114314944876241e-07, clsf_neg: 1.8684021441117693e-08, vat_loss: -1.4822326699359678e-10
            Threshold is set to 0.28413859009742737
            Min. Probailities on test set with label 1: 0.283694326877594
Testing ==> auc: 0.999928, prec: 0.2958, rec: 0.9767, F1score: 0.4541, clsf_loss: 2.6258710477122804e-06
Epoch 95 -------------------------------------------------------------------------
Training => auc: 0.995743, clsf_pos: 4.632163665974076e-07, clsf_neg: 1.8945009117032896e-08, vat_loss: 8.365139692045886e-10
            Threshold is set to 0.2835202217102051
            Min. Probailities on test set with label 1: 0.28347378969192505
Testing ==> auc: 0.999931, prec: 0.0729, rec: 0.9767, F1score: 0.1357, clsf_loss: 2.619658971525496e-06
Epoch 96 -------------------------------------------------------------------------
Training => auc: 0.999996, clsf_pos: 4.068704413384694e-07, clsf_neg: 1.8846538551997583e-08, vat_loss: 1.0597015487334716e-09
            Threshold is set to 0.2841716408729553
            Min. Probailities on test set with label 1: 0.28367581963539124
Testing ==> auc: 0.999927, prec: 0.5122, rec: 0.9767, F1score: 0.6720, clsf_loss: 2.627878075145418e-06
Epoch 97 -------------------------------------------------------------------------
Training => auc: 0.997552, clsf_pos: 4.5531547243626846e-07, clsf_neg: 1.870209409560175e-08, vat_loss: 1.0404604955382979e-09
            Threshold is set to 0.2836020290851593
            Min. Probailities on test set with label 1: 0.28368905186653137
Testing ==> auc: 0.999927, prec: 0.0420, rec: 1.0000, F1score: 0.0806, clsf_loss: 2.6234431516058976e-06
Epoch 98 -------------------------------------------------------------------------
Training => auc: 0.999579, clsf_pos: 4.4333634718896064e-07, clsf_neg: 1.9907664849938556e-08, vat_loss: 5.287431048550673e-10
            Threshold is set to 0.2837264835834503
            Min. Probailities on test set with label 1: 0.28368932008743286
Testing ==> auc: 0.999926, prec: 0.0837, rec: 0.9767, F1score: 0.1541, clsf_loss: 2.628800984894042e-06
Epoch 99 -------------------------------------------------------------------------
Training => auc: 0.999992, clsf_pos: 4.6159394173628243e-07, clsf_neg: 1.919980086029227e-08, vat_loss: -2.9118766425861864e-11
            Threshold is set to 0.283951073884964
            Min. Probailities on test set with label 1: 0.2833345830440521
Testing ==> auc: 0.999915, prec: 0.6176, rec: 0.9767, F1score: 0.7568, clsf_loss: 2.619247197799268e-06
'''
