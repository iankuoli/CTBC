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
from .optim.rangerlars import RangerLars

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR

torch.cuda.set_device(0)

# Deep SAD
deepsad_c = 0.0
deepsad_eps = 1e-6
deepsad_eta = 1.

#
# Classifier
# ---------------------
## focal loss
alpha = 1
gamma_pos = 6
gamma_neg = 1
grad_clip = 1
lambda_l1 = 0
weight_decay = 0  # lambda_l2

#
# VAT
# ---------------------
vat_xi = 1e-6
vat_eps_pos = 1000
vat_eps_neg = 0.001
vat_ip = 1

#
# Training process
# ---------------------
train_batch_size = 128
test_batch_size = 32

#
# Optimizer
# ---------------------
optim_type = 'rlars'  # ['adam', 'rlars']
learn_rate = 1e-4
adam_beta1 = 0.9
adam_beta2 = 0.999

max_epochs = 400

training_date = [(2018, 5), (2018, 6), (2018, 7), (2018, 8),
                 (2018, 9), (2018, 10), (2018, 11), (2018, 12)]

testing_date = [(2019, 1), (2019, 2), (2019, 3), (2019, 4), (2019, 5), (2019, 6)]

train_data = [np.load('../../user_data/CloudMile/data/data_{}_{}.npz'.format(year, month),
                      allow_pickle=True)
              for year, month in training_date]

test_data = [np.load('../../user_data/CloudMile/data/data_{}_{}.npz'.format(year, month),
                     allow_pickle=True)
             for year, month in testing_date]

X_train = np.concatenate([data['arr_0'] for data in train_data])
y_train = np.concatenate([data['arr_1'] for data in train_data])
training_announce = np.concatenate([data['arr_2'] for data in train_data])
training_FILTER = np.concatenate([data['arr_3'] for data in train_data])

X_test = np.concatenate([data['arr_0'] for data in test_data])
y_test = np.concatenate([data['arr_1'] for data in test_data])
testing_announce = np.concatenate([data['arr_2'] for data in test_data])
testing_FILTER = np.concatenate([data['arr_3'] for data in test_data])

# Only consider announce == 1
X_train = X_train[training_announce == 1]
y_train = y_train[training_announce == 1]
X_test = X_test[testing_announce == 1]
y_test = y_test[testing_announce == 1]

# Log Trandsform
X_train = np.dstack([np.log10(X_train[:, :, :314] + 1e-10) + 10, X_train[:, :, 314:]])
X_test = np.dstack([np.log10(X_test[:, :, :314] + 1e-10) + 10, X_test[:, :, 314:]])

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

train_data = [(X_train[i], y_train[i]) for i in range(len(X_train))]
test_data = [(X_test[i], y_test[i]) for i in range(len(X_test))]

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=test_batch_size)


class VarEncoder(nn.Module):
    def __init__(self, in_dim, convs_dim, fcs_dim, z_dim):
        super(VarEncoder, self).__init__()

        self.in_dim = in_dim
        self.conv1_dim = convs_dim[0]
        self.conv2_dim = convs_dim[1]
        self.conv3_dim = convs_dim[2]
        self.outdim_en1 = fcs_dim[0]
        self.outdim_en2 = fcs_dim[1]
        self.dim_z = z_dim

        self.model_conv = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=self.conv1_dim, kernel_size=2),
            nn.BatchNorm1d(self.conv1_dim),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv1_dim, out_channels=self.conv2_dim, kernel_size=2),
            nn.BatchNorm1d(self.conv2_dim),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv2_dim, out_channels=self.conv3_dim, kernel_size=2),
            nn.BatchNorm1d(self.conv3_dim),
            nn.ReLU(),
        )
        self.model_fc = nn.Sequential(
            # FC 1
            nn.Dropout(0.6),
            nn.Linear(in_features=self.conv3_dim, out_features=self.outdim_en1),
            nn.BatchNorm1d(self.outdim_en1),
            nn.ReLU(),
            # FC 2
            nn.Dropout(0.6),
            nn.Linear(in_features=self.outdim_en1, out_features=self.outdim_en2),
            nn.BatchNorm1d(self.outdim_en2),
            nn.ReLU(),
        )

        self.fc_zmean = nn.Linear(in_features=self.outdim_en2, out_features=self.dim_z)
        self.fc_zvar = nn.Linear(in_features=self.outdim_en2, out_features=self.dim_z)

    def forward(self, x):
        x = self.model_conv(x)
        h = self.model_fc(x.view(-1, self.in_dim * 4))
        return self.fc_zmean(h), self.fc_zvar(h)


class Decoder(nn.Module):
    def __init__(self, z_dim, convs_dim, fcs_dim, out_dim):
        super(Decoder, self).__init__()

        self.dim_z = z_dim

        self.conv1_dim = convs_dim[0]
        self.conv2_dim = convs_dim[1]
        self.outdim_en1 = fcs_dim[0]
        self.outdim_en2 = fcs_dim[1]
        self.out_dim = out_dim

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
            nn.ConvTranspose1d(in_channels=self.outdim_de2, out_channels=self.conv1_dim, kernel_size=2),
            nn.ConvTranspose1d(in_channels=self.conv1_dim, out_channels=self.conv2_dim, kernel_size=2),
            nn.ConvTranspose1d(in_channels=self.conv2_dim, out_channels=self.out_dim, kernel_size=2),
        )

    def forward(self, x):
        x = self.model_fc(x)
        return self.model_convt(x.view(-1, self.outdim_de2, 1))


class VAE(nn.Module):
    def __init__(self, z_dim, data_dim, convs_dim, fcs_dim):
        super(VAE, self).__init__()

        self.dim_z = z_dim
        self.dim_data = data_dim

        self.enc_model = VarEncoder(in_dim=self.dim_data, z_dim=self.dim_z,
                                    convs_dim=convs_dim, fcs_dim=fcs_dim)
        self.dec_model = Decoder(z_dim=self.dim_z, out_dim=self.dim_data,
                                 convs_dim=convs_dim[:-1][::-1], fcs_dim=[fcs_dim[:-1][::-1] + convs_dim[-1]])

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


#
# VAE Pretraining
# ----------------------------------------------------------------------------------------------------------------------
vae = VAE(in_dim=X_train.shape[2], out_dim=128, convs_dim=[512, 768, 512], fcs_dim=[384, 256]).cuda()


def vae_train(dataloader, clip_grad_norm=0):
    loss_batch = []

    for batch_idx, (data, target) in enumerate(dataloader):
        if data.size()[0] != dataloader.batch_size:
            continue
        data, _ = Variable(data.cuda()), Variable(target.cuda())

        # Zero the network parameter gradients
        optim_sad.zero_grad()

        # Update network parameters via backpropagation: forward + backward + optimize
        gen_data, latent = vae(data)
        dist = torch.sum((gen_data.reshape(gen_data.shape[0],-1) - data.reshape(data.shape[0],-1)) ** 2, dim=1)
        losses = dist
        loss = torch.mean(losses)
        loss.backward()

        # Gradient clipping
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=clip_grad_norm)
        optim_sad.step()

        # Record the losses
        loss_batch.append(loss)

    return sum(loss_batch) / len(loss_batch)


# Pretrain VAE
for epoch in range(max_epochs):
    vae.train()
    mse_loss = vae_train(train_dataloader, clip_grad_norm=grad_clip)
    print("Epoch: {} ==> MSE Loss = {}".format(epoch, mse_loss))


#
# Deep-SAD Training
# ----------------------------------------------------------------------------------------------------------------------
net = VAE.enc_model
optim_sad = RangerLars(net.parameters(), lr=learn_rate)


def deep_sad_train(dataloader, clip_grad_norm=0):
    loss_batch = []

    for batch_idx, (data, target) in enumerate(dataloader):
        if data.size()[0] != dataloader.batch_size:
            continue
        data, target = Variable(data.cuda()), Variable(target.cuda())
        target = target.reshape(-1, 1)

        # Zero the network parameter gradients
        optim_sad.zero_grad()

        # Update network parameters via backpropagation: forward + backward + optimize
        outputs = net(data)
        dist = torch.sum((outputs - deepsad_c) ** 2, dim=1)
        losses = torch.where(target==0, dist, deepsad_eta * ((dist + deepsad_eps) ** (-1.)))
        loss = torch.mean(losses)
        loss.backward()

        # Gradient clipping
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=clip_grad_norm)
        optim_sad.step()

        # Record the losses
        loss_batch.append(loss)

    return sum(loss_batch) / len(loss_batch)


# Train Deep-SAD
for epoch in range(max_epochs):
    net.train()
    deepsad_loss = deep_sad_train(train_dataloader, clip_grad_norm=grad_clip)
    print("Epoch: {} ==> Deep-SAD Loss = {}".format(epoch, deepsad_loss))





