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

from spectral_normalization import SpectralNorm



from weight_init import weight_init




torch.cuda.set_device(3)



## Model Declaration



## SN-GAN



class VarEncoder(nn.Module):
    def __init__(self, in_dim=256, z_dim=128):
        super(VarEncoder, self).__init__()
        
        self.in_dim = in_dim
        self.outdim_en1 = in_dim
        self.outdim_en2 = math.ceil(self.outdim_en1 / 2)
        self.dim_z = z_dim
        
        self.model_conv = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=in_dim*2, kernel_size=2),
            nn.Conv1d(in_channels=in_dim*2, out_channels=in_dim*4, kernel_size=2),
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
        )
        self.fc_zmean = nn.Linear(in_features=self.outdim_en2, out_features=self.dim_z)
        self.fc_zvar = nn.Linear(in_features=self.outdim_en2, out_features=self.dim_z)
        
    def forward(self, x):
        x = self.model_conv(x)
        h = self.model_fc(x.view(-1, self.in_dim*4))
        return self.fc_zmean(h), self.fc_zvar(h)

class Decoder(nn.Module):
    def __init__(self, z_dim=128, out_dim=256):
        super(Decoder, self).__init__()
        
        self.dim_z = z_dim
        self.outdim_de1 = math.ceil(self.dim_z * 2)
        self.outdim_de2 = math.ceil(self.outdim_de1 * 4)
        self.dim_out = out_dim
        
        self.model_fc = nn.Sequential(
            nn.Linear(in_features=self.dim_z, out_features=self.outdim_de1),
            nn.BatchNorm1d(self.outdim_de1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=self.outdim_de1, out_features=self.outdim_de2),
            nn.BatchNorm1d(self.outdim_de2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        self.model_convt = nn.Sequential(
            nn.ConvTranspose1d(in_channels=self.outdim_de2, out_channels=self.dim_out*2, kernel_size=2),
            nn.ConvTranspose1d(in_channels=self.dim_out*2, out_channels=self.dim_out, kernel_size=2),
        )
    
    def forward(self, x):
        x = self.model_fc(x)
        return self.model_convt(x.view(-1, self.outdim_de2, 1))




class Generator(nn.Module):
    def __init__(self, z_dim=128, data_dim=256, data_length=3):
        super(Generator, self).__init__()
        
        self.dim_z = z_dim
        self.dim_data = data_dim
        self.length_data = data_length
        
        self.enc_model = VarEncoder(in_dim=self.dim_data, z_dim=self.dim_z)
        self.dec_model = Decoder(z_dim=self.dim_z, out_dim=self.dim_data)
        
    def encode(self, x):
        return self.enc_model(x)
    
    def decode(self, z):
        return self.dec_model(z)
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.dim_data, self.length_data))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class Discriminator(nn.Module):
    def __init__(self, data_dim=256, data_length=3):
        super(Discriminator, self).__init__()
        
        self.dim_data = data_dim
        self.outdim1 = self.dim_data
        self.outdim2 = math.ceil(self.dim_data / 2)
        self.length_data = data_length
        
        self.model_conv = nn.Sequential(
            SpectralNorm(nn.Conv1d(in_channels=self.dim_data, out_channels=self.dim_data*2, kernel_size=2)),
            SpectralNorm(nn.Conv1d(in_channels=self.dim_data*2, out_channels=self.dim_data*4, kernel_size=2)),
        )
        
        self.model_fc = nn.Sequential(
            SpectralNorm(nn.Linear(in_features=self.dim_data*4, out_features=self.outdim1)),
            SpectralNorm(nn.BatchNorm1d(self.outdim1)),
            nn.ReLU(),
            nn.Dropout(0.4),
            SpectralNorm(nn.Linear(in_features=self.outdim1, out_features=self.outdim2)),
            SpectralNorm(nn.BatchNorm1d(self.outdim2)),
            nn.ReLU(),
            nn.Dropout(0.2),
            SpectralNorm(nn.Linear(in_features=self.outdim2, out_features=1)),
        )
        
    def forward(self, x):
        x = self.model_conv(x.view(-1, self.dim_data, self.length_data))
        return self.model_fc(x.view(-1, self.dim_data*4))




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
            F_loss, F_loss_pos, F_loss_neg




class GRU(nn.Module):
    def __init__(self, emb_size):
        super(GRU, self).__init__()
        
        self.emb_size = emb_size
        self.hid1 = 200
        self.hid2 = 100
        self.rnn = nn.GRU(self.emb_size, self.hid1, num_layers=3, 
                          bidirectional=True, batch_first=True, dropout=0.3)
        self.relu = nn.ReLU()
        
        self.out1 = nn.Sequential(
            nn.Linear(2*self.hid1, 64),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        rnn, hid = self.rnn(x)
        return self.out1(self.relu(rnn[:, -1]))
        
    def get_trainable_parameters(self):
        return (param for param in self.parameters() if param.requires_grad)




## Parameters Settings



#
# GRU
# ---------------------
## focal loss
alpha = 1e-4
gamma = 2
gamma_pos = 3
gamma_neg = 2
learn_rate = 1e-4

train_batch_size = 128
test_batch_size = 256

latent_dim = 128

gan_loss = 'wasserstein'
max_epochs = 100



## Data Preparation



data = np.load('../GRUArray_and_label.npz', allow_pickle=True)

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




## Training of GAN + GRU



### Model Initialization



#
# GRU
# ------------------------------------------------------------------------------------
classifier = GRU(X_train.shape[2]*2).cuda()
classifier.apply(weight_init)

focal_loss = FocalLoss2(alpha, gamma)
optim_clsfr = optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), 
                         lr=learn_rate)

#
# Encoder
# ------------------------------------------------------------------------------------
encoder = VarEncoder(in_dim=X_train.shape[2], z_dim=latent_dim).cuda()
optim_enc = optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), 
                       lr=learn_rate)

#
# GAN
# ------------------------------------------------------------------------------------
discriminator = Discriminator(data_dim=X_train.shape[2], 
                              data_length=X_train.shape[1]).cuda()
generator = Generator(z_dim=latent_dim, 
                      data_dim=X_train.shape[2], 
                      data_length=X_train.shape[1]).cuda()

optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), 
                        lr=learn_rate, betas=(0.0, 0.9))
optim_gen = optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), 
                        lr=learn_rate, betas=(0.0, 0.9))

scheduler_d = optim.lr_scheduler.ExponentialLR(optim_disc, gamma=0.99)
scheduler_g = optim.lr_scheduler.ExponentialLR(optim_gen, gamma=0.99)




disc_iters = 2
def train(epoch):
    label_list = []
    pred_y_list = []
    
    total_loss_batch = []
    disc_loss_batch = []
    gen_loss_batch = []
    con_loss_batch = []
    enc_loss_batch = []
    clsf_loss_batch = []
    for batch_idx, (data, target) in enumerate(train_dataloader):
        if data.size()[0] != train_batch_size:
            continue
        data, target = Variable(data.cuda()), Variable(target.cuda())
        
        # Update discriminator
        for _ in range(disc_iters):
            optim_disc.zero_grad()
            optim_gen.zero_grad()
            
            fake_data, mu, logvar = generator(data.permute(0, 2, 1))
            
            if gan_loss == 'hinge':
                disc_loss = nn.ReLU()(1.0 - discriminator(data)).mean() + \
                            nn.ReLU()(1.0 + discriminator(fake_data)).mean()
            elif gan_loss == 'wasserstein':
                disc_loss = -discriminator(data).mean() + discriminator(fake_data).mean()
            else:
                disc_loss = nn.BCEWithLogitsLoss()(discriminator(data),
                                                   Variable(torch.ones(train_batch_size,1).cuda())) + \
                            nn.BCEWithLogitsLoss()(discriminator(fake_data),
                                                   Variable(torch.ones(train_batch_size,1).cuda()))
            disc_loss.backward()
            optim_disc.step()
        
        # Update generator, encoder, and classifier
        optim_disc.zero_grad()
        optim_gen.zero_grad()
        optim_enc.zero_grad()
        optim_clsfr.zero_grad()
        
        fake_data, mu, logvar = generator(data.permute(0, 2, 1))
        fake_mu, fake_logvar = encoder(fake_data)
        pred_y = classifier(torch.cat((data, data - fake_data.permute(0, 2, 1)), 2)).squeeze(-1)
        
        if gan_loss == 'hinge' or gan_loss == 'wasserstein':
            gen_loss = -discriminator(fake_data).mean()
        else:
            gen_loss = nn.BCEWithLogitsLoss()(discriminator(fake_data),
                                              Variable(torch.ones(train_batch_size,1).cuda()))
        
        con_loss = nn.L1Loss()(fake_data.permute(0, 2, 1), data)
        enc_loss = nn.MSELoss()(generator.reparameterize(mu, logvar),
                                generator.reparameterize(fake_mu, fake_logvar))
        clsf_loss, _, _ = focal_loss(pred_y, target)
        total_loss = gen_loss + con_loss + enc_loss + clsf_loss
        total_loss.backward()
        
        optim_gen.step()
        optim_enc.step()
        optim_clsfr.step()
        
        total_loss_batch.append(total_loss)
        disc_loss_batch.append(disc_loss)
        gen_loss_batch.append(gen_loss)
        enc_loss_batch.append(enc_loss)
        con_loss_batch.append(con_loss)
        clsf_loss_batch.append(clsf_loss)
        
        label_list += list(target.cpu().detach().numpy())
        pred_y_list += list(pred_y.cpu().detach().numpy())
        
        if batch_idx % 6000 == 0:
            print('  Idx {} => disc: {:.4f}, gen:{:.4f}, con: {:.4f}, enc: {:.4f}, clsf: {}'.
                  format(batch_idx, disc_loss, gen_loss, con_loss, enc_loss, clsf_loss))
        
    
    scheduler_d.step()
    scheduler_g.step()
    
    total_loss_avg = sum(total_loss_batch) / len(total_loss_batch)
    disc_loss_avg = sum(disc_loss_batch) / len(disc_loss_batch)
    gen_loss_avg = sum(gen_loss_batch) / len(gen_loss_batch)
    con_loss_avg = sum(con_loss_batch) / len(con_loss_batch)
    enc_loss_avg = sum(enc_loss_batch) / len(enc_loss_batch)
    clsf_loss_avg = sum(clsf_loss_batch) / len(clsf_loss_batch)
    
    loss_tuple = (total_loss_avg, disc_loss_avg, gen_loss_avg, 
                  con_loss_avg, enc_loss_avg, clsf_loss_avg)
    
    return  np.array(label_list), np.array(pred_y_list), loss_tuple




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
        fake_data, mu, logvar = generator(data.permute(0, 2, 1))
        pred_y = classifier(torch.cat((data, data - fake_data.permute(0, 2, 1)), 2)).squeeze(-1)
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
          
    discriminator.train(mode=True)
    generator.train(mode=True)
    encoder.train(mode=True)
    classifier.train(mode=True)
    
    label_train, pred_y_train, loss_train = train(epoch)
    
    auc_train = roc_auc_score(label_train, pred_y_train)
    train_history_loss.append(loss_train)
    train_history_auc.append(auc_train)
    
    print('Training => auc:{:.6f}, total: {:.4f}, clsf: {}, disc: {:.4f}, gen: {:.4f}'.
          format(auc_train, loss_train[0], loss_train[-1], loss_train[1], loss_train[2]))
    
    if epoch % 1 == 0:
        #
        # Testing
        # ------------------------------------------------------------------------------------        
        thres = np.min(pred_y_train[label_train==1])
        print("            Threshold is set to {}".format(thres))
        
        with torch.no_grad():
            classifier.eval()
            label_test, pred_y_test, clsf_loss_test, _, _ = infer(test_dataloader)    
        
        auc = roc_auc_score(label_test, pred_y_test)
        
        print("            Min. Probailities on test set with label 1: {}".
              format(np.min(pred_y_test[label_test==1])))
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
               'saved_models/conv1d_sngan_clsfr-auc{:.6f}-thres{:.4f}'.format(auc_train, thres))






Parameter Setting ----------------------------------------------------------------------
Model = GAN-SN + GRU
graph_emdeding = Count_larger_8K
alpha = 0.0001
gamma_pos = 3
gamma_neg = 2
learn_rate = 0.0001
train_batch_size = 128
test_batch_size = 256
max_epochs = 100

Epoch 0 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2720, gen:0.0999, con: 0.0058, enc: 0.0035, clsf: 8.853024979771362e-09
  Idx 6000 => disc: -0.2478, gen:0.1045, con: 0.0066, enc: 0.0024, clsf: 1.0092017888041482e-08
Training => auc:0.515440, total: 0.1141, clsf: 6.357982584859201e-08, disc: -0.2790, gen: 0.1053
Threshold is set to 0.005555183161050081
Min. Probailities on test set with label 1: 0.04389047622680664
Testing ==> auc: 0.981164, prec: 0.0002, rec: 1.0000, F1score: 0.0003, clsf_loss: 5.6120320834907034e-08
Epoch 1 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2574, gen:0.1068, con: 0.0039, enc: 0.0019, clsf: 1.0429933361422172e-08
  Idx 6000 => disc: -0.2731, gen:0.0744, con: 0.0085, enc: 0.0023, clsf: 5.135817549017929e-09
Training => auc:0.523547, total: 0.1076, clsf: 6.26666079028837e-08, disc: -0.2771, gen: 0.0987
Threshold is set to 0.010763832367956638
Min. Probailities on test set with label 1: 0.034982431679964066
Testing ==> auc: 0.980549, prec: 0.0002, rec: 1.0000, F1score: 0.0003, clsf_loss: 5.604237784950783e-08
Epoch 2 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2816, gen:0.1015, con: 0.0135, enc: 0.0019, clsf: 2.2214537693798775e-06
  Idx 6000 => disc: -0.3001, gen:0.0545, con: 0.0055, enc: 0.0035, clsf: 6.047249367924223e-09
Training => auc:0.468355, total: 0.1028, clsf: 6.358704496278733e-08, disc: -0.2847, gen: 0.0929
Threshold is set to 0.008289066143333912
Min. Probailities on test set with label 1: 0.04118262603878975
Testing ==> auc: 0.980503, prec: 0.0002, rec: 1.0000, F1score: 0.0003, clsf_loss: 5.527061830434832e-08
Epoch 3 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2554, gen:0.0746, con: 0.0106, enc: 0.0016, clsf: 8.7068929843781e-09
  Idx 6000 => disc: -0.2143, gen:-0.0165, con: 0.0096, enc: 0.0019, clsf: 9.784002052981577e-09
Training => auc:0.478667, total: 0.0891, clsf: 6.232878746459392e-08, disc: -0.2961, gen: 0.0779
Threshold is set to 0.01083077397197485
Min. Probailities on test set with label 1: 0.043478824198246
Testing ==> auc: 0.980414, prec: 0.0002, rec: 1.0000, F1score: 0.0003, clsf_loss: 5.52442607215653e-08
Epoch 4 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2832, gen:0.0777, con: 0.0051, enc: 0.0016, clsf: 9.165242786934868e-09
  Idx 6000 => disc: -0.3121, gen:0.1496, con: 0.0069, enc: 0.0024, clsf: 8.526263250985266e-09
Training => auc:0.549073, total: 0.1019, clsf: 5.993977225671188e-08, disc: -0.2929, gen: 0.0918
Threshold is set to 0.015619231387972832
Min. Probailities on test set with label 1: 0.042182691395282745
Testing ==> auc: 0.980588, prec: 0.0002, rec: 1.0000, F1score: 0.0003, clsf_loss: 5.3366523644626795e-08
Epoch 5 -------------------------------------------------------------------------
  Idx 0 => disc: -0.3247, gen:0.0577, con: 0.0048, enc: 0.0021, clsf: 2.834920451277867e-06
  Idx 6000 => disc: -0.2468, gen:0.0764, con: 0.0087, enc: 0.0016, clsf: 9.078402030127108e-09
Training => auc:0.618617, total: 0.0963, clsf: 5.798989732852533e-08, disc: -0.2936, gen: 0.0859
Threshold is set to 0.011934945359826088
Min. Probailities on test set with label 1: 0.05617516115307808
Testing ==> auc: 0.980646, prec: 0.0002, rec: 1.0000, F1score: 0.0003, clsf_loss: 5.469861719120672e-08
Epoch 6 -------------------------------------------------------------------------
  Idx 0 => disc: -0.3289, gen:0.1463, con: 0.0049, enc: 0.0016, clsf: 1.6103731681482714e-08
  Idx 6000 => disc: -0.3139, gen:0.0725, con: 0.0062, enc: 0.0011, clsf: 5.2114588200424805e-09
Training => auc:0.810881, total: 0.1058, clsf: 5.0971589615755875e-08, disc: -0.2911, gen: 0.0959
Threshold is set to 0.010353625752031803
Min. Probailities on test set with label 1: 0.021841226145625114
Testing ==> auc: 0.984612, prec: 0.0003, rec: 1.0000, F1score: 0.0005, clsf_loss: 4.213039161982124e-08
Epoch 7 -------------------------------------------------------------------------
  Idx 0 => disc: -0.1672, gen:-0.0742, con: 0.0064, enc: 0.0012, clsf: 1.8494387132861334e-09
  Idx 6000 => disc: -0.2913, gen:0.1142, con: 0.0227, enc: 0.0013, clsf: 1.5368057937337198e-09
Training => auc:0.932921, total: 0.0986, clsf: 4.132504471954235e-08, disc: -0.2927, gen: 0.0879
Threshold is set to 0.008730429224669933
Min. Probailities on test set with label 1: 0.02742912992835045
Testing ==> auc: 0.985527, prec: 0.0003, rec: 1.0000, F1score: 0.0006, clsf_loss: 2.7336202634842266e-08
Epoch 8 -------------------------------------------------------------------------
  Idx 0 => disc: -0.3143, gen:0.1325, con: 0.0119, enc: 0.0013, clsf: 9.206774898018466e-09
  Idx 6000 => disc: -0.2659, gen:0.1210, con: 0.0098, enc: 0.0017, clsf: 8.055408007301423e-10
Training => auc:0.989075, total: 0.0964, clsf: 2.6256532947854794e-08, disc: -0.2906, gen: 0.0862
Threshold is set to 0.008416840806603432
Min. Probailities on test set with label 1: 0.021426210179924965
Testing ==> auc: 0.998074, prec: 0.0006, rec: 1.0000, F1score: 0.0012, clsf_loss: 1.932387760916754e-08
Epoch 9 -------------------------------------------------------------------------
  Idx 0 => disc: -0.3489, gen:0.1075, con: 0.0125, enc: 0.0011, clsf: 5.122509638688655e-10
  Idx 6000 => disc: -0.1199, gen:-0.0288, con: 0.0060, enc: 0.0007, clsf: 1.2996269616039058e-09
Training => auc:0.995748, total: 0.1125, clsf: 2.136208543390694e-08, disc: -0.2902, gen: 0.1034
Threshold is set to 0.012390382587909698
Min. Probailities on test set with label 1: 0.019287271425127983
Testing ==> auc: 0.998617, prec: 0.0232, rec: 1.0000, F1score: 0.0453, clsf_loss: 1.803787874621321e-08
Epoch 10 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2760, gen:0.0922, con: 0.0060, enc: 0.0027, clsf: 3.880662458044526e-09
  Idx 6000 => disc: -0.2704, gen:0.0701, con: 0.0114, enc: 0.0022, clsf: 1.051116069183955e-10
Training => auc:0.997153, total: 0.1237, clsf: 1.9746165591527642e-08, disc: -0.2915, gen: 0.1146
Threshold is set to 0.01602664217352867
Min. Probailities on test set with label 1: 0.1460122913122177
Testing ==> auc: 0.999075, prec: 0.0257, rec: 1.0000, F1score: 0.0501, clsf_loss: 1.5778580220171534e-08
Epoch 11 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2596, gen:0.0910, con: 0.0116, enc: 0.0009, clsf: 1.681156192034905e-08
  Idx 6000 => disc: -0.3208, gen:0.1343, con: 0.0109, enc: 0.0012, clsf: 2.5398219985484083e-11
Training => auc:0.998372, total: 0.1046, clsf: 1.7274087937835247e-08, disc: -0.2879, gen: 0.0949
Threshold is set to 0.0257944967597723
Min. Probailities on test set with label 1: 0.1690797358751297
Testing ==> auc: 0.999348, prec: 0.0348, rec: 1.0000, F1score: 0.0672, clsf_loss: 1.3837620826961938e-08
Epoch 12 -------------------------------------------------------------------------
  Idx 0 => disc: -0.3156, gen:0.1366, con: 0.0111, enc: 0.0010, clsf: 1.5120674157209635e-10
  Idx 6000 => disc: -0.2763, gen:0.1104, con: 0.0056, enc: 0.0007, clsf: 3.4660236486461216e-11
Training => auc:0.993079, total: 0.1115, clsf: 1.5836924660561635e-08, disc: -0.2871, gen: 0.1026
Threshold is set to 0.001085580326616764
Min. Probailities on test set with label 1: 0.26195764541625977
Testing ==> auc: 0.999828, prec: 0.0003, rec: 1.0000, F1score: 0.0007, clsf_loss: 1.1788742426688259e-08
Epoch 13 -------------------------------------------------------------------------
  Idx 0 => disc: -0.3154, gen:0.1010, con: 0.0096, enc: 0.0007, clsf: 5.47412462581498e-11
  Idx 6000 => disc: -0.3088, gen:0.1342, con: 0.0067, enc: 0.0012, clsf: 3.790006530834944e-08
Training => auc:0.999453, total: 0.1130, clsf: 1.1100108388006902e-08, disc: -0.2910, gen: 0.1035
Threshold is set to 0.02057061158120632
Min. Probailities on test set with label 1: 0.12291444838047028
Testing ==> auc: 0.999931, prec: 0.0749, rec: 1.0000, F1score: 0.1394, clsf_loss: 5.980425044072035e-09
Epoch 14 -------------------------------------------------------------------------
  Idx 0 => disc: -0.3058, gen:0.1362, con: 0.0065, enc: 0.0014, clsf: 4.9712348820785124e-11
  Idx 6000 => disc: -0.2850, gen:0.1029, con: 0.0075, enc: 0.0006, clsf: 1.9291443442703837e-11
Training => auc:0.995455, total: 0.1076, clsf: 7.999396700597572e-09, disc: -0.2899, gen: 0.0984
Threshold is set to 0.00027128701913170516
Min. Probailities on test set with label 1: 0.013699479401111603
Testing ==> auc: 0.999936, prec: 0.0003, rec: 1.0000, F1score: 0.0007, clsf_loss: 6.4311351799517524e-09
Epoch 15 -------------------------------------------------------------------------
  Idx 0 => disc: -0.3426, gen:0.1478, con: 0.0077, enc: 0.0006, clsf: 1.53943177649829e-11
  Idx 6000 => disc: -0.1789, gen:-0.0465, con: 0.0064, enc: 0.0012, clsf: 1.1102972707899283e-10
Training => auc:0.999759, total: 0.0939, clsf: 7.931668655203339e-09, disc: -0.2915, gen: 0.0832
Threshold is set to 0.025236625224351883
Min. Probailities on test set with label 1: 0.3992239236831665
Testing ==> auc: 0.999942, prec: 0.0403, rec: 1.0000, F1score: 0.0774, clsf_loss: 1.4636525769162745e-08
