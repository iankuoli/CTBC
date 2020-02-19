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

### Model Initialization

SN-GAN


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




class ConvClsfr(nn.Module):
    def __init__(self, in_dim=256, out_dim=1):
        super(ConvClsfr, self).__init__()
        
        self.in_dim = in_dim
        self.outdim_en1 = in_dim
        self.outdim_en2 = math.ceil(self.outdim_en1 / 2)
        self.out_dim = out_dim
        
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
            nn.Linear(in_features=self.outdim_en2, out_features=self.out_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.model_conv(x)
        return self.model_fc(x.view(-1, self.in_dim*4))



class GRUClsfr(nn.Module):
    def __init__(self, emb_size):
        super(GRUClsfr, self).__init__()
        
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




## Training of GAN + Conv1D


### Model Initialization



#
# Conv1D as classifier
# ------------------------------------------------------------------------------------
classifier = ConvClsfr(in_dim=X_train.shape[2]*2, out_dim=1).cuda()
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
        
        data = data.permute(0, 2, 1)
        fake_data, mu, logvar = generator(data)
        fake_mu, fake_logvar = encoder(fake_data)
        pred_y = classifier(torch.cat((data, data - fake_data), 1)).squeeze(-1)
        
        if gan_loss == 'hinge' or gan_loss == 'wasserstein':
            gen_loss = -discriminator(fake_data).mean()
        else:
            gen_loss = nn.BCEWithLogitsLoss()(discriminator(fake_data),
                                              Variable(torch.ones(train_batch_size,1).cuda()))
        
        con_loss = nn.L1Loss()(fake_data, data)
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
        data = data.permute(0, 2, 1)
        fake_data, mu, logvar = generator(data)
        pred_y = classifier(torch.cat((data, data - fake_data), 1)).squeeze(-1)
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
Model = SN-GAN + Conv1D
conv1d use activation = False
graph_emdeding = Count_larger_8K
alpha = 0.0001
gamma_pos = 3
gamma_neg = 2
learn_rate = 0.0001
train_batch_size = 128
test_batch_size = 256
max_epochs = 100
gan_loss = 'wasserstein'

Epoch 0 -------------------------------------------------------------------------
  Idx 0 => disc: -0.0080, gen:0.0616, con: 0.0843, enc: 2.1906, clsf: 1.6466945453430526e-05
  Idx 6000 => disc: -0.3365, gen:0.0323, con: 0.0128, enc: 0.0038, clsf: 1.4169342366088244e-09
Training => auc:0.988092, total: 0.1374, clsf: 1.551186983306252e-07, disc: -0.3380, gen: 0.0376
Threshold is set to 0.014666968956589699
Min. Probailities on test set with label 1: 0.08196160942316055
Testing ==> auc: 0.999980, prec: 0.0002, rec: 1.0000, F1score: 0.0003, clsf_loss: 3.7219427539980643e-09
Epoch 1 -------------------------------------------------------------------------
  Idx 0 => disc: -0.3811, gen:0.0704, con: 0.0146, enc: 0.0040, clsf: 1.2393386317199884e-09
  Idx 6000 => disc: -0.3013, gen:0.0449, con: 0.0077, enc: 0.0039, clsf: 1.6166824823304893e-10
Training => auc:0.999938, total: 0.0748, clsf: 1.8166775861416795e-09, disc: -0.3186, gen: 0.0620
Threshold is set to 0.0411192812025547
Min. Probailities on test set with label 1: 0.006401661783456802
Testing ==> auc: 0.976634, prec: 1.0000, rec: 0.9302, F1score: 0.9639, clsf_loss: 5.014924031598866e-09
Epoch 2 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2995, gen:0.0458, con: 0.0074, enc: 0.0026, clsf: 1.5341956871584017e-10
  Idx 6000 => disc: -0.2911, gen:0.0383, con: 0.0067, enc: 0.0032, clsf: 5.2359189900430536e-11
Training => auc:0.992509, total: 0.0788, clsf: 2.000029475368592e-09, disc: -0.2949, gen: 0.0688
Threshold is set to 0.0030768890865147114
Min. Probailities on test set with label 1: 0.0045060222037136555
Testing ==> auc: 0.976919, prec: 0.0002, rec: 1.0000, F1score: 0.0003, clsf_loss: 5.210417874934592e-09
Epoch 3 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2944, gen:0.0367, con: 0.0077, enc: 0.0025, clsf: 5.925257506866899e-11
  Idx 6000 => disc: -0.3389, gen:0.0605, con: 0.0122, enc: 0.0023, clsf: 9.061101174934194e-11
Training => auc:0.999930, total: 0.0681, clsf: 2.052084280279587e-09, disc: -0.2952, gen: 0.0579
Threshold is set to 0.03255137801170349
Min. Probailities on test set with label 1: 0.05251000076532364
Testing ==> auc: 0.999965, prec: 0.0252, rec: 1.0000, F1score: 0.0493, clsf_loss: 2.72648392751762e-09
Epoch 4 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2576, gen:0.0374, con: 0.0078, enc: 0.0027, clsf: 1.734851151624639e-10
  Idx 6000 => disc: -0.2986, gen:0.0765, con: 0.0108, enc: 0.0023, clsf: 8.852144017801322e-11
Training => auc:0.993179, total: 0.0491, clsf: 2.773500540342866e-09, disc: -0.3028, gen: 0.0374
Threshold is set to 0.0025293168146163225
Min. Probailities on test set with label 1: 0.027735915035009384
Testing ==> auc: 0.999920, prec: 0.0002, rec: 1.0000, F1score: 0.0003, clsf_loss: 2.96024604828915e-09
Epoch 5 -------------------------------------------------------------------------
  Idx 0 => disc: -0.3371, gen:0.0703, con: 0.0043, enc: 0.0022, clsf: 3.215658295196988e-11
  Idx 6000 => disc: -0.3266, gen:-0.0000, con: 0.0096, enc: 0.0016, clsf: 2.034882679025074e-11
Training => auc:0.999997, total: 0.0524, clsf: 8.532114459391948e-10, disc: -0.3021, gen: 0.0413
Threshold is set to 0.09472698718309402
Min. Probailities on test set with label 1: 0.04106837883591652
Testing ==> auc: 0.999978, prec: 0.6833, rec: 0.9535, F1score: 0.7961, clsf_loss: 2.5869266728761886e-09
Epoch 6 -------------------------------------------------------------------------
  Idx 0 => disc: -0.3289, gen:-0.0530, con: 0.0184, enc: 0.0017, clsf: 4.498403385899685e-11
  Idx 6000 => disc: -0.3043, gen:0.0308, con: 0.0323, enc: 0.0073, clsf: 1.0669478495151097e-10
Training => auc:0.993594, total: 0.0633, clsf: 1.6386322299055678e-09, disc: -0.3008, gen: 0.0526
Threshold is set to 0.0015824956353753805
Min. Probailities on test set with label 1: 0.01360747218132019
Testing ==> auc: 0.999916, prec: 0.0002, rec: 1.0000, F1score: 0.0004, clsf_loss: 3.3180078684580394e-09
Epoch 7 -------------------------------------------------------------------------
  Idx 0 => disc: -0.3086, gen:0.0072, con: 0.0059, enc: 0.0020, clsf: 5.133769104270769e-11
  Idx 6000 => disc: -0.2994, gen:0.0275, con: 0.0067, enc: 0.0010, clsf: 2.7534948279783755e-11
Training => auc:0.999871, total: 0.0562, clsf: 1.5046167645138553e-09, disc: -0.3050, gen: 0.0451
Threshold is set to 0.015371098183095455
Min. Probailities on test set with label 1: 0.02205863781273365
Testing ==> auc: 0.999929, prec: 0.0360, rec: 1.0000, F1score: 0.0695, clsf_loss: 2.763088868817931e-09
Epoch 8 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2703, gen:0.0296, con: 0.0038, enc: 0.0014, clsf: 8.413882437996456e-12
  Idx 6000 => disc: -0.3258, gen:0.0447, con: 0.0054, enc: 0.0009, clsf: 3.2398382587839336e-11
Training => auc:0.999512, total: 0.0584, clsf: 2.567498436079063e-09, disc: -0.2993, gen: 0.0486
Threshold is set to 0.01371445506811142
Min. Probailities on test set with label 1: 0.06208653375506401
Testing ==> auc: 0.999958, prec: 0.0006, rec: 1.0000, F1score: 0.0012, clsf_loss: 2.4707247359145867e-09
Epoch 9 -------------------------------------------------------------------------
  Idx 0 => disc: -0.3119, gen:0.0285, con: 0.0045, enc: 0.0009, clsf: 1.4882166332608193e-10
  Idx 6000 => disc: -0.3140, gen:0.0706, con: 0.0098, enc: 0.0015, clsf: 2.8357023326708308e-11
Training => auc:0.997374, total: 0.0492, clsf: 1.5373777806360067e-09, disc: -0.3000, gen: 0.0396
Threshold is set to 0.003952203784137964
Min. Probailities on test set with label 1: 0.09374828636646271
Testing ==> auc: 0.999966, prec: 0.0002, rec: 1.0000, F1score: 0.0004, clsf_loss: 2.0155357383089267e-09
Epoch 10 -------------------------------------------------------------------------
  Idx 0 => disc: -0.3250, gen:0.0724, con: 0.0068, enc: 0.0006, clsf: 1.3810097163058366e-11
  Idx 6000 => disc: -0.2742, gen:-0.0018, con: 0.0213, enc: 0.0010, clsf: 5.349716850067132e-11
Training => auc:0.999909, total: 0.0539, clsf: 9.691398794586803e-10, disc: -0.3045, gen: 0.0433
Threshold is set to 0.014454730786383152
Min. Probailities on test set with label 1: 0.032423995435237885
Testing ==> auc: 0.999897, prec: 0.0016, rec: 1.0000, F1score: 0.0033, clsf_loss: 3.000189874313719e-09
Epoch 11 -------------------------------------------------------------------------
  Idx 0 => disc: -0.3081, gen:0.0481, con: 0.0186, enc: 0.0011, clsf: 1.7370939756067294e-11
  Idx 6000 => disc: -0.3364, gen:0.1053, con: 0.0069, enc: 0.0008, clsf: 8.228605297144753e-12
Training => auc:0.999098, total: 0.0553, clsf: 1.2114018677067406e-09, disc: -0.3024, gen: 0.0446
Threshold is set to 0.005669959355145693
Min. Probailities on test set with label 1: 0.02124895341694355
Testing ==> auc: 0.999886, prec: 0.0007, rec: 1.0000, F1score: 0.0014, clsf_loss: 2.661595832620378e-09
Epoch 12 -------------------------------------------------------------------------
  Idx 0 => disc: -0.3086, gen:0.0864, con: 0.0053, enc: 0.0020, clsf: 1.2114186390133064e-11
  Idx 6000 => disc: -0.2707, gen:-0.0559, con: 0.0074, enc: 0.0014, clsf: 4.201031449796444e-12
Training => auc:0.999918, total: 0.0560, clsf: 1.2852802155904897e-09, disc: -0.2945, gen: 0.0465
Threshold is set to 0.01721234805881977
Min. Probailities on test set with label 1: 0.0036804352421313524
Testing ==> auc: 0.990761, prec: 0.0324, rec: 0.9767, F1score: 0.0627, clsf_loss: 4.110077611585439e-09
Epoch 13 -------------------------------------------------------------------------
  Idx 0 => disc: -0.3466, gen:0.0794, con: 0.0078, enc: 0.0022, clsf: 6.379458159649909e-12
  Idx 6000 => disc: -0.3138, gen:0.0535, con: 0.0143, enc: 0.0021, clsf: 1.5227931762784586e-11
Training => auc:0.999983, total: 0.0532, clsf: 1.4761708522215145e-09, disc: -0.2955, gen: 0.0429
Threshold is set to 0.06013071537017822
Min. Probailities on test set with label 1: 0.027895918115973473
Testing ==> auc: 0.999941, prec: 0.3443, rec: 0.9767, F1score: 0.5091, clsf_loss: 2.6713535827838086e-09
Epoch 14 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2892, gen:0.0322, con: 0.0033, enc: 0.0012, clsf: 1.5977630460004e-12
  Idx 6000 => disc: -0.2991, gen:0.0595, con: 0.0101, enc: 0.0009, clsf: 3.1022320129414638e-12
Training => auc:0.999971, total: 0.0534, clsf: 1.1803880095584418e-09, disc: -0.2849, gen: 0.0439
Threshold is set to 0.022752225399017334
Min. Probailities on test set with label 1: 0.01995360106229782
Testing ==> auc: 0.999955, prec: 0.0942, rec: 0.9767, F1score: 0.1718, clsf_loss: 3.088490574398861e-09
Epoch 15 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2868, gen:0.0477, con: 0.0034, enc: 0.0016, clsf: 1.8520332836780007e-12
  Idx 6000 => disc: -0.3030, gen:0.0388, con: 0.0035, enc: 0.0007, clsf: 5.361495136746441e-10
Training => auc:0.999980, total: 0.0497, clsf: 1.1202468952475897e-09, disc: -0.2949, gen: 0.0384
Threshold is set to 0.028337974101305008
Min. Probailities on test set with label 1: 0.035651493817567825
Testing ==> auc: 0.999968, prec: 0.0846, rec: 1.0000, F1score: 0.1561, clsf_loss: 2.627908779473387e-09
Epoch 16 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2969, gen:0.0360, con: 0.0032, enc: 0.0007, clsf: 9.78194630331386e-13
  Idx 6000 => disc: -0.3152, gen:0.0513, con: 0.0140, enc: 0.0015, clsf: 3.4053763281471916e-11
Training => auc:0.999986, total: 0.0574, clsf: 8.740430046394465e-10, disc: -0.2857, gen: 0.0471
Threshold is set to 0.029938306659460068
Min. Probailities on test set with label 1: 0.00392843596637249
Testing ==> auc: 0.992630, prec: 0.1673, rec: 0.9767, F1score: 0.2857, clsf_loss: 3.307008444863868e-09
Epoch 17 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2263, gen:0.0032, con: 0.0099, enc: 0.0010, clsf: 8.209943141990195e-12
  Idx 6000 => disc: -0.2792, gen:0.0496, con: 0.0148, enc: 0.0012, clsf: 3.767519282660281e-11
Training => auc:0.996834, total: 0.0632, clsf: 1.8674457535894362e-09, disc: -0.2852, gen: 0.0527
Threshold is set to 0.0019357851706445217
Min. Probailities on test set with label 1: 0.027057074010372162
Testing ==> auc: 0.999933, prec: 0.0002, rec: 1.0000, F1score: 0.0004, clsf_loss: 2.6173720968358793e-09
Epoch 18 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2630, gen:0.0482, con: 0.0101, enc: 0.0019, clsf: 6.679850683005695e-12
  Idx 6000 => disc: -0.2880, gen:0.0656, con: 0.0124, enc: 0.0010, clsf: 6.602361886376418e-12
Training => auc:0.999815, total: 0.0663, clsf: 1.0141573136834836e-09, disc: -0.2873, gen: 0.0554
Threshold is set to 0.008815745823085308
Min. Probailities on test set with label 1: 0.016596360132098198
Testing ==> auc: 0.999936, prec: 0.0076, rec: 1.0000, F1score: 0.0151, clsf_loss: 3.1035629621811722e-09
Epoch 19 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2634, gen:0.0265, con: 0.0030, enc: 0.0020, clsf: 1.412546945731008e-12
  Idx 6000 => disc: -0.2700, gen:0.0381, con: 0.0104, enc: 0.0010, clsf: 4.095061529457711e-12
Training => auc:0.999991, total: 0.0753, clsf: 9.134834555446503e-10, disc: -0.2838, gen: 0.0649
Threshold is set to 0.037877704948186874
Min. Probailities on test set with label 1: 0.0275456253439188
Testing ==> auc: 0.999959, prec: 0.1458, rec: 0.9767, F1score: 0.2538, clsf_loss: 2.483325989288687e-09
Epoch 20 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2887, gen:0.0462, con: 0.0036, enc: 0.0009, clsf: 3.656386391576172e-12
  Idx 6000 => disc: -0.2567, gen:0.0538, con: 0.0031, enc: 0.0023, clsf: 3.8143948177978004e-13
Training => auc:0.999923, total: 0.0792, clsf: 1.1585508108424847e-09, disc: -0.2842, gen: 0.0686
Threshold is set to 0.014243646524846554
Min. Probailities on test set with label 1: 0.02171177975833416
Testing ==> auc: 0.999935, prec: 0.0375, rec: 1.0000, F1score: 0.0722, clsf_loss: 2.9083295771670237e-09
Epoch 21 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2510, gen:0.0514, con: 0.0041, enc: 0.0025, clsf: 4.135062084409391e-12
  Idx 6000 => disc: -0.2894, gen:0.1038, con: 0.0124, enc: 0.0014, clsf: 9.916255316877454e-12
Training => auc:0.999984, total: 0.0825, clsf: 8.529995043637939e-10, disc: -0.2870, gen: 0.0714
Threshold is set to 0.02582971379160881
Min. Probailities on test set with label 1: 0.0541963092982769
Testing ==> auc: 0.999968, prec: 0.0464, rec: 1.0000, F1score: 0.0887, clsf_loss: 2.1735242494713702e-09
Epoch 22 -------------------------------------------------------------------------
  Idx 0 => disc: -0.3316, gen:0.1020, con: 0.0104, enc: 0.0034, clsf: 3.7665912056006334e-12
  Idx 6000 => disc: -0.3316, gen:0.0604, con: 0.0097, enc: 0.0023, clsf: 8.812429085069962e-12
Training => auc:0.999983, total: 0.0685, clsf: 1.030959984049673e-09, disc: -0.3089, gen: 0.0562
Threshold is set to 0.03320743888616562
Min. Probailities on test set with label 1: 0.03350435942411423
Testing ==> auc: 0.999948, prec: 0.0779, rec: 1.0000, F1score: 0.1445, clsf_loss: 2.7407849323424216e-09
Epoch 23 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2958, gen:0.0384, con: 0.0242, enc: 0.0056, clsf: 3.655935797153287e-12
  Idx 6000 => disc: -0.2760, gen:-0.0200, con: 0.0093, enc: 0.0016, clsf: 4.415155741011034e-12
Training => auc:0.999955, total: 0.0255, clsf: 1.7918948547190894e-09, disc: -0.2780, gen: 0.0155
Threshold is set to 0.03143256902694702
Min. Probailities on test set with label 1: 0.050799861550331116
Testing ==> auc: 0.999958, prec: 0.0493, rec: 1.0000, F1score: 0.0940, clsf_loss: 2.391616904517946e-09
Epoch 24 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2621, gen:-0.0075, con: 0.0140, enc: 0.0022, clsf: 2.2320706047351813e-11
  Idx 6000 => disc: -0.2689, gen:-0.0128, con: 0.0053, enc: 0.0007, clsf: 3.634528875778864e-12
Training => auc:0.999972, total: 0.0240, clsf: 9.500451536581522e-10, disc: -0.2754, gen: 0.0141
Threshold is set to 0.024238068610429764
Min. Probailities on test set with label 1: 0.04562569037079811
Testing ==> auc: 0.999936, prec: 0.0401, rec: 1.0000, F1score: 0.0771, clsf_loss: 2.2833621660112158e-09
Epoch 25 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2169, gen:0.0601, con: 0.0054, enc: 0.0020, clsf: 6.475858261092982e-13
  Idx 6000 => disc: -0.3261, gen:0.0461, con: 0.0053, enc: 0.0020, clsf: 2.85818742805366e-12
Training => auc:0.999986, total: 0.0329, clsf: 1.1253906695429805e-09, disc: -0.2805, gen: 0.0223
Threshold is set to 0.03773588314652443
Min. Probailities on test set with label 1: 0.06307105720043182
Testing ==> auc: 0.999914, prec: 0.0180, rec: 1.0000, F1score: 0.0353, clsf_loss: 2.408918842178309e-09
Epoch 26 -------------------------------------------------------------------------
  Idx 0 => disc: -0.3891, gen:0.1318, con: 0.0047, enc: 0.0020, clsf: 6.514181555283827e-13
  Idx 6000 => disc: -0.2908, gen:0.0478, con: 0.0037, enc: 0.0040, clsf: 1.1064938028326754e-12
Training => auc:0.999970, total: 0.0524, clsf: 1.713420849647207e-09, disc: -0.2788, gen: 0.0417
Threshold is set to 0.03403173387050629
Min. Probailities on test set with label 1: 0.01700502634048462
Testing ==> auc: 0.999943, prec: 0.2471, rec: 0.9767, F1score: 0.3944, clsf_loss: 3.203036724741537e-09
Epoch 27 -------------------------------------------------------------------------
  Idx 0 => disc: -0.3081, gen:0.0300, con: 0.0030, enc: 0.0027, clsf: 6.294173515719592e-12
  Idx 6000 => disc: -0.2633, gen:0.0573, con: 0.0031, enc: 0.0028, clsf: 1.5486547591189725e-12
Training => auc:0.999807, total: 0.0431, clsf: 1.76976810983831e-09, disc: -0.2826, gen: 0.0325
Threshold is set to 0.006786954589188099
Min. Probailities on test set with label 1: 0.02151421830058098
Testing ==> auc: 0.999960, prec: 0.0082, rec: 1.0000, F1score: 0.0162, clsf_loss: 3.158238781608702e-09
Epoch 28 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2593, gen:-0.0019, con: 0.0126, enc: 0.0010, clsf: 1.0046353506043548e-11
  Idx 6000 => disc: -0.3052, gen:0.0237, con: 0.0030, enc: 0.0007, clsf: 3.104898716604909e-12
Training => auc:0.999826, total: 0.0358, clsf: 1.1703462643453122e-09, disc: -0.2782, gen: 0.0254
Threshold is set to 0.013469092547893524
Min. Probailities on test set with label 1: 0.06823524087667465
Testing ==> auc: 0.999959, prec: 0.0311, rec: 1.0000, F1score: 0.0604, clsf_loss: 2.065240645166e-09
Epoch 29 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2987, gen:0.0408, con: 0.0030, enc: 0.0008, clsf: 1.8799406475977776e-12
  Idx 6000 => disc: -0.2980, gen:0.0557, con: 0.0034, enc: 0.0011, clsf: 9.026341306339614e-12
Training => auc:0.999960, total: 0.0462, clsf: 1.1571085201111941e-09, disc: -0.2866, gen: 0.0352
Threshold is set to 0.01701340638101101
Min. Probailities on test set with label 1: 0.1013328805565834
Testing ==> auc: 0.999939, prec: 0.0248, rec: 1.0000, F1score: 0.0484, clsf_loss: 3.0641582604573614e-09
Epoch 30 -------------------------------------------------------------------------
  Idx 0 => disc: -0.3140, gen:0.0572, con: 0.0032, enc: 0.0026, clsf: 2.569290558085413e-09
  Idx 6000 => disc: -0.2287, gen:-0.0421, con: 0.0063, enc: 0.0010, clsf: 3.678428547003454e-13
Training => auc:0.999981, total: 0.0403, clsf: 8.233375647925811e-10, disc: -0.2827, gen: 0.0298
Threshold is set to 0.033425286412239075
Min. Probailities on test set with label 1: 0.04021822661161423
Testing ==> auc: 0.999951, prec: 0.0613, rec: 1.0000, F1score: 0.1154, clsf_loss: 2.298206736028874e-09
Epoch 31 -------------------------------------------------------------------------
  Idx 0 => disc: -0.3048, gen:0.0540, con: 0.0032, enc: 0.0018, clsf: 2.70016132933662e-13
  Idx 6000 => disc: -0.3462, gen:0.0370, con: 0.0062, enc: 0.0014, clsf: 4.1596907125393545e-13
Training => auc:0.999971, total: 0.0414, clsf: 8.627638048430697e-10, disc: -0.2831, gen: 0.0309
Threshold is set to 0.017599821090698242
Min. Probailities on test set with label 1: 0.04899977892637253
Testing ==> auc: 0.999954, prec: 0.0388, rec: 1.0000, F1score: 0.0747, clsf_loss: 2.320192482585526e-09
Epoch 32 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2368, gen:0.0163, con: 0.0053, enc: 0.0074, clsf: 5.456839949520564e-13
  Idx 6000 => disc: -0.2818, gen:0.1203, con: 0.0041, enc: 0.0023, clsf: 2.4797464429071603e-12
Training => auc:0.999986, total: 0.0428, clsf: 7.801854717826018e-10, disc: -0.2825, gen: 0.0321
Threshold is set to 0.03478453308343887
Min. Probailities on test set with label 1: 0.023618297651410103
Testing ==> auc: 0.999905, prec: 0.0972, rec: 0.9767, F1score: 0.1768, clsf_loss: 3.0365325809356136e-09
Epoch 33 -------------------------------------------------------------------------
  Idx 0 => disc: -0.3045, gen:-0.0035, con: 0.0147, enc: 0.0019, clsf: 5.000580852176917e-11
  Idx 6000 => disc: -0.3107, gen:0.0480, con: 0.0034, enc: 0.0018, clsf: 2.669378173364434e-12
Training => auc:0.999967, total: 0.0331, clsf: 1.0500725844408976e-09, disc: -0.2832, gen: 0.0223
Threshold is set to 0.02191043831408024
Min. Probailities on test set with label 1: 0.0311005599796772
Testing ==> auc: 0.999892, prec: 0.0252, rec: 1.0000, F1score: 0.0492, clsf_loss: 2.5029678329957505e-09
Epoch 34 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2694, gen:-0.0828, con: 0.0064, enc: 0.0043, clsf: 3.1303516094365047e-13
  Idx 6000 => disc: -0.2568, gen:0.0068, con: 0.0111, enc: 0.0011, clsf: 1.3013279689944035e-11
Training => auc:0.999978, total: 0.0441, clsf: 8.781300131488479e-10, disc: -0.2963, gen: 0.0328
Threshold is set to 0.027909770607948303
Min. Probailities on test set with label 1: 0.03195098787546158
Testing ==> auc: 0.999933, prec: 0.0817, rec: 1.0000, F1score: 0.1511, clsf_loss: 3.0790967553429027e-09
Epoch 35 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2983, gen:0.0535, con: 0.0032, enc: 0.0023, clsf: 4.77223482953093e-10
  Idx 6000 => disc: -0.3071, gen:0.0556, con: 0.0029, enc: 0.0006, clsf: 4.62989726852242e-13
Training => auc:0.999976, total: 0.0323, clsf: 7.436366522561855e-10, disc: -0.2834, gen: 0.0212
Threshold is set to 0.017581140622496605
Min. Probailities on test set with label 1: 0.03148841857910156
Testing ==> auc: 0.999939, prec: 0.0366, rec: 1.0000, F1score: 0.0706, clsf_loss: 2.610422100701726e-09
Epoch 36 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2396, gen:0.0056, con: 0.0163, enc: 0.0008, clsf: 3.507096522914477e-12
  Idx 6000 => disc: -0.2715, gen:0.0382, con: 0.0091, enc: 0.0021, clsf: 2.9309896523721513e-12
Training => auc:0.999973, total: 0.0287, clsf: 9.972881409581191e-10, disc: -0.2826, gen: 0.0182
Threshold is set to 0.01683441363275051
Min. Probailities on test set with label 1: 0.06263794749975204
Testing ==> auc: 0.999902, prec: 0.0196, rec: 1.0000, F1score: 0.0385, clsf_loss: 2.699436452147097e-09
Epoch 37 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2447, gen:0.0096, con: 0.0094, enc: 0.0023, clsf: 3.4695149531138725e-11
  Idx 6000 => disc: -0.3179, gen:0.0413, con: 0.0031, enc: 0.0012, clsf: 9.583150245573435e-13
Training => auc:0.999988, total: 0.0253, clsf: 9.139501933042027e-10, disc: -0.2774, gen: 0.0145
Threshold is set to 0.04459788277745247
Min. Probailities on test set with label 1: 0.030696600675582886
Testing ==> auc: 0.999933, prec: 0.0800, rec: 0.9767, F1score: 0.1479, clsf_loss: 2.5910327217104623e-09
Epoch 38 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2658, gen:-0.0005, con: 0.0143, enc: 0.0028, clsf: 6.441047695204816e-11
  Idx 6000 => disc: -0.3011, gen:0.0526, con: 0.0031, enc: 0.0030, clsf: 2.7497117430219653e-11
Training => auc:0.999969, total: 0.0377, clsf: 9.92320225989829e-10, disc: -0.2870, gen: 0.0267
Threshold is set to 0.02247343212366104
Min. Probailities on test set with label 1: 0.029702747240662575
Testing ==> auc: 0.999947, prec: 0.0535, rec: 1.0000, F1score: 0.1017, clsf_loss: 2.8146651676053125e-09
Epoch 39 -------------------------------------------------------------------------
  Idx 0 => disc: -0.3096, gen:0.1112, con: 0.0037, enc: 0.0010, clsf: 2.608644366058205e-13
  Idx 6000 => disc: -0.2083, gen:-0.0493, con: 0.0071, enc: 0.0009, clsf: 5.526092604346555e-12
Training => auc:0.999975, total: 0.0477, clsf: 9.329733652307937e-10, disc: -0.2968, gen: 0.0360
Threshold is set to 0.021231798455119133
Min. Probailities on test set with label 1: 0.03315621241927147
Testing ==> auc: 0.999941, prec: 0.0418, rec: 1.0000, F1score: 0.0802, clsf_loss: 2.7889333065189703e-09
Epoch 40 -------------------------------------------------------------------------
  Idx 0 => disc: -0.3169, gen:0.0213, con: 0.0110, enc: 0.0017, clsf: 1.763409947577732e-12
  Idx 6000 => disc: -0.3155, gen:0.0436, con: 0.0034, enc: 0.0017, clsf: 4.639056066374492e-12
Training => auc:0.999977, total: 0.0378, clsf: 1.1093602703127203e-09, disc: -0.2893, gen: 0.0256
Threshold is set to 0.037889569997787476
Min. Probailities on test set with label 1: 0.045987796038389206
Testing ==> auc: 0.999944, prec: 0.0647, rec: 1.0000, F1score: 0.1215, clsf_loss: 2.5375379575365287e-09
Epoch 41 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2029, gen:-0.0815, con: 0.0073, enc: 0.0016, clsf: 1.539447458398513e-10
  Idx 6000 => disc: -0.2531, gen:-0.0140, con: 0.0123, enc: 0.0012, clsf: 2.7068061334012405e-12
Training => auc:0.999994, total: 0.0349, clsf: 7.492890752303083e-10, disc: -0.2941, gen: 0.0230
Threshold is set to 0.06126074120402336
Min. Probailities on test set with label 1: 0.03545388579368591
Testing ==> auc: 0.999922, prec: 0.1123, rec: 0.9767, F1score: 0.2014, clsf_loss: 2.3989581432459772e-09
Epoch 42 -------------------------------------------------------------------------
  Idx 0 => disc: -0.3280, gen:0.0648, con: 0.0032, enc: 0.0007, clsf: 7.454401679263256e-11
  Idx 6000 => disc: -0.1629, gen:0.0358, con: 0.0054, enc: 0.0059, clsf: 2.314687504167967e-12
Training => auc:0.999928, total: 0.0165, clsf: 9.77940173285674e-10, disc: -0.2797, gen: 0.0058
Threshold is set to 0.010666867718100548
Min. Probailities on test set with label 1: 0.028667284175753593
Testing ==> auc: 0.999832, prec: 0.0043, rec: 1.0000, F1score: 0.0086, clsf_loss: 2.6221911308965673e-09
Epoch 43 -------------------------------------------------------------------------
  Idx 0 => disc: -0.3119, gen:0.0472, con: 0.0038, enc: 0.0022, clsf: 7.467118590120947e-10
  Idx 6000 => disc: -0.3011, gen:0.0345, con: 0.0033, enc: 0.0015, clsf: 7.660640242816708e-13
Training => auc:0.999984, total: 0.0334, clsf: 9.50782008679596e-10, disc: -0.2913, gen: 0.0220
Threshold is set to 0.03686026483774185
Min. Probailities on test set with label 1: 0.02231675386428833
Testing ==> auc: 0.999933, prec: 0.1333, rec: 0.9767, F1score: 0.2346, clsf_loss: 3.0380262749929443e-09
Epoch 44 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2885, gen:0.0083, con: 0.0082, enc: 0.0010, clsf: 1.8189907913246373e-11
  Idx 6000 => disc: -0.3195, gen:-0.0165, con: 0.0094, enc: 0.0016, clsf: 5.720365500727631e-11
Training => auc:0.999978, total: 0.0477, clsf: 7.433061388617546e-10, disc: -0.3081, gen: 0.0364
Threshold is set to 0.029470335692167282
Min. Probailities on test set with label 1: 0.012664323672652245
Testing ==> auc: 0.998948, prec: 0.0189, rec: 0.9767, F1score: 0.0370, clsf_loss: 3.7067595659578956e-09
Epoch 45 -------------------------------------------------------------------------
  Idx 0 => disc: -0.3133, gen:0.0439, con: 0.0098, enc: 0.0011, clsf: 5.983560001643351e-12
  Idx 6000 => disc: -0.3223, gen:0.0500, con: 0.0033, enc: 0.0027, clsf: 4.322939033650497e-13
Training => auc:0.999940, total: 0.0549, clsf: 1.3983967317443557e-09, disc: -0.2957, gen: 0.0442
Threshold is set to 0.01700127311050892
Min. Probailities on test set with label 1: 0.031287990510463715
Testing ==> auc: 0.999941, prec: 0.0407, rec: 1.0000, F1score: 0.0782, clsf_loss: 2.799394938080013e-09
Epoch 46 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2878, gen:0.0542, con: 0.0031, enc: 0.0013, clsf: 8.396313700732849e-13
  Idx 6000 => disc: -0.2942, gen:0.0373, con: 0.0158, enc: 0.0015, clsf: 1.6772014007661318e-13
Training => auc:0.999993, total: 0.0608, clsf: 7.49440953740077e-10, disc: -0.2939, gen: 0.0517
Threshold is set to 0.05672413855791092
Min. Probailities on test set with label 1: 0.029724018648266792
Testing ==> auc: 0.999931, prec: 0.1803, rec: 0.9767, F1score: 0.3043, clsf_loss: 2.773153706669973e-09
Epoch 47 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2424, gen:-0.0832, con: 0.0070, enc: 0.0011, clsf: 3.310914042931046e-11
  Idx 6000 => disc: -0.3027, gen:0.0576, con: 0.0033, enc: 0.0009, clsf: 3.3883413436130994e-11
Training => auc:0.999996, total: 0.0450, clsf: 5.815044556989335e-10, disc: -0.2944, gen: 0.0330
Threshold is set to 0.05670393630862236
Min. Probailities on test set with label 1: 0.04101778194308281
Testing ==> auc: 0.999935, prec: 0.0800, rec: 0.9767, F1score: 0.1479, clsf_loss: 2.307890989428074e-09
Epoch 48 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2545, gen:-0.0813, con: 0.0072, enc: 0.0013, clsf: 5.5029574647091906e-11
  Idx 6000 => disc: -0.3103, gen:0.0695, con: 0.0116, enc: 0.0013, clsf: 1.3042930291362609e-08
Training => auc:0.996713, total: 0.0525, clsf: 1.346956213232886e-09, disc: -0.2974, gen: 0.0406
Threshold is set to 0.0008478241506963968
Min. Probailities on test set with label 1: 0.0755714476108551
Testing ==> auc: 0.999936, prec: 0.0003, rec: 1.0000, F1score: 0.0006, clsf_loss: 3.0771833969822637e-09
Epoch 49 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2715, gen:0.0623, con: 0.0030, enc: 0.0014, clsf: 1.7124881157223881e-12
  Idx 6000 => disc: -0.3304, gen:0.0315, con: 0.0062, enc: 0.0028, clsf: 1.1400373917461182e-12
Training => auc:0.999990, total: 0.0514, clsf: 5.431192162674847e-10, disc: -0.3052, gen: 0.0410
Threshold is set to 0.04210522770881653
Min. Probailities on test set with label 1: 0.04865115135908127
Testing ==> auc: 0.999934, prec: 0.0497, rec: 1.0000, F1score: 0.0947, clsf_loss: 2.2304107449855337e-09
Epoch 50 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2833, gen:0.0154, con: 0.0124, enc: 0.0021, clsf: 4.886760700956172e-12
  Idx 6000 => disc: -0.3056, gen:0.1166, con: 0.0076, enc: 0.0014, clsf: 2.2440840418552765e-12
Training => auc:0.992902, total: 0.0567, clsf: 3.909725876383163e-09, disc: -0.3160, gen: 0.0439
Threshold is set to 0.0006543079507537186
Min. Probailities on test set with label 1: 0.05617281794548035
Testing ==> auc: 0.999932, prec: 0.0002, rec: 1.0000, F1score: 0.0005, clsf_loss: 2.199571857985916e-09
Epoch 51 -------------------------------------------------------------------------
  Idx 0 => disc: -0.3017, gen:0.0615, con: 0.0054, enc: 0.0012, clsf: 8.680683949435775e-11
  Idx 6000 => disc: -0.3119, gen:0.0458, con: 0.0036, enc: 0.0005, clsf: 2.5881003069827546e-12
Training => auc:0.999996, total: 0.0764, clsf: 8.044216959213202e-10, disc: -0.3061, gen: 0.0645
Threshold is set to 0.0736226886510849
Min. Probailities on test set with label 1: 0.03179511800408363
Testing ==> auc: 0.999903, prec: 0.2135, rec: 0.9535, F1score: 0.3489, clsf_loss: 2.9259286105087767e-09
Epoch 52 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2748, gen:0.0803, con: 0.0040, enc: 0.0009, clsf: 7.599734101575162e-13
  Idx 6000 => disc: -0.3165, gen:0.0481, con: 0.0034, enc: 0.0005, clsf: 4.5245288093388736e-13
Training => auc:0.999952, total: 0.0689, clsf: 1.0061892430357489e-09, disc: -0.3051, gen: 0.0571
Threshold is set to 0.016224602237343788
Min. Probailities on test set with label 1: 0.029123995453119278
Testing ==> auc: 0.999934, prec: 0.0408, rec: 1.0000, F1score: 0.0785, clsf_loss: 2.9038575988238335e-09
Epoch 53 -------------------------------------------------------------------------
  Idx 0 => disc: -0.3051, gen:0.0132, con: 0.0092, enc: 0.0009, clsf: 1.4025553721902506e-12
  Idx 6000 => disc: -0.3177, gen:0.0749, con: 0.0057, enc: 0.0017, clsf: 7.061956575071804e-10
Training => auc:0.999994, total: 0.0658, clsf: 6.09880923541084e-10, disc: -0.3000, gen: 0.0546
Threshold is set to 0.05852002650499344
Min. Probailities on test set with label 1: 0.02786765992641449
Testing ==> auc: 0.999931, prec: 0.2993, rec: 0.9535, F1score: 0.4556, clsf_loss: 2.975615975842061e-09
Epoch 54 -------------------------------------------------------------------------
  Idx 0 => disc: -0.3075, gen:0.0432, con: 0.0092, enc: 0.0009, clsf: 5.990349275647455e-13
  Idx 6000 => disc: -0.2810, gen:0.0316, con: 0.0082, enc: 0.0004, clsf: 8.031296981625413e-12
Training => auc:0.999992, total: 0.0648, clsf: 6.949765207764358e-10, disc: -0.2998, gen: 0.0548
Threshold is set to 0.04040808975696564
Min. Probailities on test set with label 1: 0.03785339370369911
Testing ==> auc: 0.999938, prec: 0.0674, rec: 0.9767, F1score: 0.1261, clsf_loss: 2.3915638358573688e-09
Epoch 55 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2944, gen:0.0452, con: 0.0131, enc: 0.0013, clsf: 5.813454925784889e-11
  Idx 6000 => disc: -0.3026, gen:-0.0718, con: 0.0041, enc: 0.0007, clsf: 5.4881765798597826e-11
Training => auc:0.999985, total: 0.0638, clsf: 8.878846546878094e-10, disc: -0.2954, gen: 0.0538
Threshold is set to 0.03424299508333206
Min. Probailities on test set with label 1: 0.02242613025009632
Testing ==> auc: 0.999929, prec: 0.0840, rec: 0.9767, F1score: 0.1547, clsf_loss: 2.9518074651235793e-09
Epoch 56 -------------------------------------------------------------------------
  Idx 0 => disc: -0.3110, gen:0.0402, con: 0.0103, enc: 0.0012, clsf: 2.375613811569921e-12
  Idx 6000 => disc: -0.2822, gen:0.0490, con: 0.0033, enc: 0.0004, clsf: 8.738360512614007e-14
Training => auc:0.999991, total: 0.0732, clsf: 8.764450276643743e-10, disc: -0.3015, gen: 0.0638
Threshold is set to 0.04365558177232742
Min. Probailities on test set with label 1: 0.03212961554527283
Testing ==> auc: 0.999929, prec: 0.0806, rec: 0.9767, F1score: 0.1489, clsf_loss: 2.500617712897224e-09
Epoch 57 -------------------------------------------------------------------------
  Idx 0 => disc: -0.3048, gen:0.0397, con: 0.0115, enc: 0.0007, clsf: 9.173573359277931e-13
  Idx 6000 => disc: -0.2772, gen:-0.0275, con: 0.0042, enc: 0.0006, clsf: 3.619679697034611e-13
Training => auc:0.999760, total: 0.0651, clsf: 1.3786854990982533e-09, disc: -0.3012, gen: 0.0552
Threshold is set to 0.0055067953653633595
Min. Probailities on test set with label 1: 0.03786047175526619
Testing ==> auc: 0.999966, prec: 0.0199, rec: 1.0000, F1score: 0.0390, clsf_loss: 2.6210142944904646e-09
Epoch 58 -------------------------------------------------------------------------
  Idx 0 => disc: -0.3078, gen:0.0680, con: 0.0102, enc: 0.0015, clsf: 1.1332063759583733e-12
  Idx 6000 => disc: -0.2820, gen:0.0511, con: 0.0129, enc: 0.0007, clsf: 1.5697840249906392e-10
Training => auc:0.999972, total: 0.0601, clsf: 1.1005758526749787e-09, disc: -0.2940, gen: 0.0499
Threshold is set to 0.018437344580888748
Min. Probailities on test set with label 1: 0.043437615036964417
Testing ==> auc: 0.999908, prec: 0.0071, rec: 1.0000, F1score: 0.0141, clsf_loss: 2.5688862148598446e-09
Epoch 59 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2511, gen:0.0781, con: 0.0052, enc: 0.0008, clsf: 2.0634857711421262e-10
  Idx 6000 => disc: -0.2720, gen:0.0754, con: 0.0081, enc: 0.0007, clsf: 1.4994334441610635e-13
Training => auc:0.999987, total: 0.0742, clsf: 8.841619103527876e-10, disc: -0.2992, gen: 0.0640
Threshold is set to 0.05012670159339905
Min. Probailities on test set with label 1: 0.02895534224808216
Testing ==> auc: 0.999922, prec: 0.1265, rec: 0.9767, F1score: 0.2240, clsf_loss: 2.945579780089247e-09
Epoch 60 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2826, gen:0.0661, con: 0.0032, enc: 0.0013, clsf: 7.819867149649884e-12
  Idx 6000 => disc: -0.2757, gen:0.0175, con: 0.0112, enc: 0.0007, clsf: 6.790841760445643e-13
Training => auc:0.999902, total: 0.0617, clsf: 1.746215616549307e-09, disc: -0.3020, gen: 0.0529
Threshold is set to 0.009860497899353504
Min. Probailities on test set with label 1: 0.0240887813270092
Testing ==> auc: 0.999928, prec: 0.0350, rec: 1.0000, F1score: 0.0676, clsf_loss: 2.7989581763421256e-09
Epoch 61 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2844, gen:0.0334, con: 0.0142, enc: 0.0009, clsf: 3.4795389226155438e-12
  Idx 6000 => disc: -0.3812, gen:0.0412, con: 0.0102, enc: 0.0010, clsf: 2.2321704760182998e-13
Training => auc:0.999970, total: 0.0437, clsf: 1.0208193179650493e-09, disc: -0.2956, gen: 0.0340
Threshold is set to 0.01937376894056797
Min. Probailities on test set with label 1: 0.024983566254377365
Testing ==> auc: 0.999934, prec: 0.0500, rec: 1.0000, F1score: 0.0952, clsf_loss: 2.5133466419191564e-09
Epoch 62 -------------------------------------------------------------------------
  Idx 0 => disc: -0.3075, gen:0.0393, con: 0.0101, enc: 0.0004, clsf: 1.3743871221227194e-13
  Idx 6000 => disc: -0.2925, gen:0.0394, con: 0.0104, enc: 0.0008, clsf: 2.3650673433572855e-12
Training => auc:0.999983, total: 0.0471, clsf: 7.209842722843973e-10, disc: -0.2950, gen: 0.0376
Threshold is set to 0.027392854914069176
Min. Probailities on test set with label 1: 0.025256454944610596
Testing ==> auc: 0.999929, prec: 0.0605, rec: 0.9767, F1score: 0.1140, clsf_loss: 6.0586549111008026e-09
Epoch 63 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2895, gen:-0.0094, con: 0.0091, enc: 0.0005, clsf: 2.275932534882147e-13
  Idx 6000 => disc: -0.2941, gen:0.0379, con: 0.0128, enc: 0.0007, clsf: 2.501739440607942e-11
Training => auc:0.999979, total: 0.0493, clsf: 8.101316284481186e-10, disc: -0.3002, gen: 0.0386
Threshold is set to 0.030460722744464874
Min. Probailities on test set with label 1: 0.021375341340899467
Testing ==> auc: 0.999782, prec: 0.0596, rec: 0.9535, F1score: 0.1122, clsf_loss: 3.3526279530349257e-09
Epoch 64 -------------------------------------------------------------------------
  Idx 0 => disc: -0.3238, gen:0.1072, con: 0.0089, enc: 0.0013, clsf: 1.7589643283577594e-11
  Idx 6000 => disc: -0.3350, gen:0.0594, con: 0.0034, enc: 0.0004, clsf: 3.1052013716413585e-13
Training => auc:0.999324, total: 0.0461, clsf: 1.0515704973457218e-09, disc: -0.3104, gen: 0.0345
Threshold is set to 0.0038951190654188395
Min. Probailities on test set with label 1: 0.021092111244797707
Testing ==> auc: 0.999942, prec: 0.0013, rec: 1.0000, F1score: 0.0025, clsf_loss: 3.2197615684737002e-09
Epoch 65 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2815, gen:0.0171, con: 0.0157, enc: 0.0030, clsf: 1.2797283198417997e-11
  Idx 6000 => disc: -0.2780, gen:-0.0208, con: 0.0108, enc: 0.0010, clsf: 1.5707755734138684e-13
Training => auc:0.999981, total: 0.0326, clsf: 1.0455464272141057e-09, disc: -0.3026, gen: 0.0212
Threshold is set to 0.03276277333498001
Min. Probailities on test set with label 1: 0.01927190274000168
Testing ==> auc: 0.999924, prec: 0.1235, rec: 0.9767, F1score: 0.2193, clsf_loss: 3.206412690914817e-09
Epoch 66 -------------------------------------------------------------------------
  Idx 0 => disc: -0.3743, gen:0.0875, con: 0.0077, enc: 0.0028, clsf: 1.1763413854115612e-10
  Idx 6000 => disc: -0.3231, gen:0.0134, con: 0.0236, enc: 0.0087, clsf: 5.6293934791451505e-11
Training => auc:0.993980, total: 0.0212, clsf: 2.218044414803444e-09, disc: -0.2985, gen: 0.0108
Threshold is set to 0.00010754351387731731
Min. Probailities on test set with label 1: 0.029865209013223648
Testing ==> auc: 0.999928, prec: 0.0002, rec: 1.0000, F1score: 0.0004, clsf_loss: 2.896093143078815e-09
Epoch 67 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2147, gen:-0.0820, con: 0.0068, enc: 0.0017, clsf: 1.1877400105109182e-12
  Idx 6000 => disc: -0.3080, gen:-0.0280, con: 0.0097, enc: 0.0007, clsf: 2.571196944592402e-12
Training => auc:0.999994, total: 0.0134, clsf: 6.31591112743024e-10, disc: -0.2917, gen: 0.0031
Threshold is set to 0.04529271647334099
Min. Probailities on test set with label 1: 0.04497368261218071
Testing ==> auc: 0.999938, prec: 0.0664, rec: 0.9767, F1score: 0.1243, clsf_loss: 2.420311062678593e-09
Epoch 68 -------------------------------------------------------------------------
  Idx 0 => disc: -0.3397, gen:0.0411, con: 0.0035, enc: 0.0033, clsf: 4.150627758159331e-12
  Idx 6000 => disc: -0.3059, gen:0.0186, con: 0.0032, enc: 0.0019, clsf: 1.6254009314702056e-13
Training => auc:0.999987, total: 0.0169, clsf: 6.531700180723021e-10, disc: -0.2965, gen: 0.0063
Threshold is set to 0.04184264689683914
Min. Probailities on test set with label 1: 0.03454790636897087
Testing ==> auc: 0.999950, prec: 0.1217, rec: 0.9767, F1score: 0.2165, clsf_loss: 2.9015971847456967e-09
Epoch 69 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2671, gen:-0.0320, con: 0.0139, enc: 0.0013, clsf: 5.418701511800128e-12
  Idx 6000 => disc: -0.3625, gen:-0.0648, con: 0.0074, enc: 0.0006, clsf: 1.3022671482842973e-11
Training => auc:0.999974, total: 0.0112, clsf: 6.527826612590104e-10, disc: -0.2877, gen: 0.0014
Threshold is set to 0.02565314993262291
Min. Probailities on test set with label 1: 0.02539335936307907
Testing ==> auc: 0.999941, prec: 0.0664, rec: 0.9767, F1score: 0.1243, clsf_loss: 3.1208744477595474e-09
Epoch 70 -------------------------------------------------------------------------
  Idx 0 => disc: -0.3405, gen:0.0484, con: 0.0035, enc: 0.0043, clsf: 7.253126593255443e-13
  Idx 6000 => disc: -0.2712, gen:-0.0379, con: 0.0129, enc: 0.0009, clsf: 5.905825307961354e-12
Training => auc:0.999968, total: 0.0208, clsf: 1.021961848479691e-09, disc: -0.2997, gen: 0.0106
Threshold is set to 0.03405822068452835
Min. Probailities on test set with label 1: 0.037445276975631714
Testing ==> auc: 0.999932, prec: 0.0546, rec: 1.0000, F1score: 0.1035, clsf_loss: 2.5278084070379236e-09
Epoch 71 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2176, gen:-0.0785, con: 0.0074, enc: 0.0004, clsf: 8.549070197420849e-13
  Idx 6000 => disc: -0.2361, gen:-0.0718, con: 0.0146, enc: 0.0007, clsf: 5.855303763548092e-13
Training => auc:0.999950, total: 0.0122, clsf: 9.338477768849884e-10, disc: -0.3017, gen: 0.0026
Threshold is set to 0.019451836124062538
Min. Probailities on test set with label 1: 0.025504879653453827
Testing ==> auc: 0.999907, prec: 0.0326, rec: 1.0000, F1score: 0.0632, clsf_loss: 2.5591011532100083e-09
Epoch 72 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2721, gen:0.0159, con: 0.0115, enc: 0.0009, clsf: 3.364333464594971e-11
  Idx 6000 => disc: -0.2900, gen:-0.0356, con: 0.0106, enc: 0.0008, clsf: 1.5212699806463337e-12
Training => auc:0.999994, total: 0.0129, clsf: 7.199156826231956e-10, disc: -0.3014, gen: 0.0015
Threshold is set to 0.05352722108364105
Min. Probailities on test set with label 1: 0.0988372340798378
Testing ==> auc: 0.999942, prec: 0.0420, rec: 1.0000, F1score: 0.0807, clsf_loss: 4.2299408420376494e-09
Epoch 73 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2572, gen:-0.0739, con: 0.0126, enc: 0.0009, clsf: 4.1721414517637e-11
  Idx 6000 => disc: -0.1769, gen:0.0047, con: 0.0065, enc: 0.0010, clsf: 1.0056881136243437e-13
Training => auc:0.999979, total: -0.0045, clsf: 6.007557229459337e-10, disc: -0.2902, gen: -0.0138
Threshold is set to 0.029467131942510605
Min. Probailities on test set with label 1: 0.03309264034032822
Testing ==> auc: 0.999926, prec: 0.0482, rec: 1.0000, F1score: 0.0919, clsf_loss: 2.8977920063510965e-09
Epoch 74 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2044, gen:-0.1408, con: 0.0061, enc: 0.0017, clsf: 4.478765719840433e-13
  Idx 6000 => disc: -0.3483, gen:0.0492, con: 0.0030, enc: 0.0011, clsf: 1.2569901174618037e-13
Training => auc:0.999979, total: -0.0006, clsf: 7.434568516373474e-10, disc: -0.2913, gen: -0.0099
Threshold is set to 0.024845441803336143
Min. Probailities on test set with label 1: 0.07243001461029053
Testing ==> auc: 0.999938, prec: 0.0380, rec: 1.0000, F1score: 0.0732, clsf_loss: 2.520067710065632e-09
Epoch 75 -------------------------------------------------------------------------
  Idx 0 => disc: -0.3201, gen:0.0448, con: 0.0040, enc: 0.0005, clsf: 5.010142691344588e-13
  Idx 6000 => disc: -0.3353, gen:0.0348, con: 0.0031, enc: 0.0010, clsf: 9.189489447170018e-11
Training => auc:0.999987, total: -0.0079, clsf: 5.465654040470724e-10, disc: -0.2853, gen: -0.0179
Threshold is set to 0.03659490868449211
Min. Probailities on test set with label 1: 0.0274253748357296
Testing ==> auc: 0.999906, prec: 0.0991, rec: 0.9767, F1score: 0.1799, clsf_loss: 3.414390103984033e-09
Epoch 76 -------------------------------------------------------------------------
  Idx 0 => disc: -0.3330, gen:0.0500, con: 0.0033, enc: 0.0004, clsf: 2.2906798528322947e-13
  Idx 6000 => disc: -0.2479, gen:-0.1007, con: 0.0068, enc: 0.0016, clsf: 2.0842717742691752e-12
Training => auc:0.999982, total: 0.0023, clsf: 5.642024625274189e-10, disc: -0.2866, gen: -0.0079
Threshold is set to 0.027551375329494476
Min. Probailities on test set with label 1: 0.030366728082299232
Testing ==> auc: 0.999927, prec: 0.0500, rec: 1.0000, F1score: 0.0952, clsf_loss: 2.8486792924553583e-09
Epoch 77 -------------------------------------------------------------------------
  Idx 0 => disc: -0.3391, gen:0.0515, con: 0.0041, enc: 0.0012, clsf: 2.214898457784248e-13
  Idx 6000 => disc: -0.3132, gen:0.0183, con: 0.0034, enc: 0.0007, clsf: 2.4869197345444953e-14
Training => auc:0.999990, total: 0.0005, clsf: 6.255565510038252e-10, disc: -0.2836, gen: -0.0089
Threshold is set to 0.04076407104730606
Min. Probailities on test set with label 1: 0.02952655218541622
Testing ==> auc: 0.999934, prec: 0.0877, rec: 0.9767, F1score: 0.1609, clsf_loss: 2.8763089687799948e-09
Epoch 78 -------------------------------------------------------------------------
  Idx 0 => disc: -0.2714, gen:-0.0022, con: 0.0064, enc: 0.0011, clsf: 6.866936760972081e-14
  Idx 6000 => disc: -0.1776, gen:0.0437, con: 0.0066, enc: 0.0011, clsf: 3.934923675959212e-13
Training => auc:0.999993, total: 0.0280, clsf: 7.524149081561404e-10, disc: -0.2977, gen: 0.0174
Threshold is set to 0.056968722492456436
Min. Probailities on test set with label 1: 0.03583858162164688
Testing ==> auc: 0.999946, prec: 0.1743, rec: 0.9767, F1score: 0.2958, clsf_loss: 2.5795792168992193e-09
