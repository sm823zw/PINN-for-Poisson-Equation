import math
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda:0')

background = '#D7E5E5'
mpl.rcParams['font.family']= 'sans-serif'
mpl.rcParams['font.size'] = 12
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['legend.title_fontsize'] = 10
mpl.rcParams['savefig.facecolor']= 'white'
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['axes.labelweight'] = 'heavy'

path = '../Data/data_y_100_100_58_vgs_100_50_100.csv'
data_f = pd.read_csv(path)

path = '../Data/data_bc1_vgs_100_50_100.csv'
data_bc1 = pd.read_csv(path)

path = '../Data/data_bc2_vgs_100_50_100.csv'
data_bc2 = pd.read_csv(path)

batch_size = 4096
n_batches = len(data_f.index)/batch_size

X = data_f.drop('psi', axis = 1).values.astype(np.float32)
y = data_f['psi'].values.astype(np.float32)

train = torch.tensor(X).to(device)
train_target = torch.tensor(y).to(device)
train_tensor = TensorDataset(train, train_target)
train_loader = DataLoader(dataset = train_tensor, batch_size = batch_size, shuffle = True)

mean = torch.mean(train, axis=0)
std = torch.std(train, axis=0)

X_bc1 = data_bc1.drop('psi', axis = 1).values.astype(np.float32)
y_bc1 = data_bc1['psi'].values.astype(np.float32)
train_bc1 = torch.tensor(X_bc1, requires_grad=True).to(device)
train_target_bc1 = torch.tensor(y_bc1).to(device)

X_bc2 = data_bc2.drop('psi', axis = 1).values.astype(np.float32)
y_bc2 = data_bc2['psi'].values.astype(np.float32)
train_bc2 = torch.tensor(X_bc2, requires_grad=True).to(device)
train_target_bc2 = torch.tensor(y_bc2).to(device)


class NN_SC1(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_layers):
        super().__init__()
        self.alpha = nn.parameter.Parameter(data=torch.ones(1))
        self.fcs = nn.Sequential(*[
            nn.Linear(n_input, n_hidden),
            nn.Tanh(),
        ])
        self.fch = nn.Sequential(*[
            nn.Sequential(*[
                nn.Linear(n_hidden, n_hidden),
                nn.Tanh(),
            ]) for _ in range(n_layers - 1)
        ])
        self.fce = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = (x-mean)/std
        x = self.alpha*x
        x = torch.exp(x)
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

class NN_SC2(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_layers):
        super().__init__()
        self.alpha = nn.parameter.Parameter(data=-1*torch.ones(1))
        self.fcs = nn.Sequential(*[
            nn.Linear(n_input, n_hidden),
            nn.Tanh(),
        ])
        self.fch = nn.Sequential(*[
            nn.Sequential(*[
                nn.Linear(n_hidden, n_hidden),
                nn.Tanh(),
            ]) for _ in range(n_layers - 1)
        ])
        self.fce = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = (x-mean)/std
        x = self.alpha*x
        x = torch.exp(x)
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

model = NN_SC1(2, 1, 20, 4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epoch_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 250, gamma=0.97)

N_A = 1e22
t_ox = 1e-9
t_si = 4e-7
epsilon_0 = 8.85418781e-12
epsilon_si = epsilon_0*11.9
epsilon_sio2 = epsilon_0*3.9
delta_psi_MS = 0.21
psi_t = 26e-3
n_i = 1e16
psi_F = psi_t*math.log(N_A/n_i)
q = 1.6e-19
p_o = N_A
n_o = n_i**2/N_A
A = q*N_A/epsilon_si
B = epsilon_sio2/(t_ox*epsilon_si)
C = np.exp(-2*psi_F/psi_t)

n_epochs = 50000
loss = nn.MSELoss(reduction='sum')
min_loss = 9999

loss_tot, loss_1, loss_2, loss_3, loss_4 = [], [], [], [], []

for i in range(n_epochs):
    mse, mse_1, mse_2, mse_3, mse_4 = 0, 0, 0, 0, 0
    for X, psi_true in train_loader:
        optimizer.zero_grad()
        
        X.requires_grad = True
        psi_pred = model(X).squeeze()
        l1 = loss(psi_pred, psi_true)
        
        d_psi = torch.autograd.grad(psi_pred, X, torch.ones_like(psi_pred), create_graph=True)[0]
        d_psi_2 = torch.autograd.grad(d_psi, X, torch.ones_like(d_psi), create_graph=True)[0]
        d_psi_2 = d_psi_2[:, 0]
        f = d_psi_2 / A + (torch.exp(-psi_true/psi_t) - 1 - C*(torch.exp(psi_true/psi_t) - 1))
        l2 = loss(f, torch.zeros_like(f))
        
        psi_bc1_pred = model(train_bc1).squeeze()
        d_psi_bc1 = torch.autograd.grad(psi_bc1_pred, train_bc1, torch.ones_like(psi_bc1_pred), create_graph=True)[0]
        d_psi_bc1 = d_psi_bc1[:, 0]

        bc1 = d_psi_bc1 / B + (train_bc1[:, 1] - train_target_bc1)
        l3 = loss(bc1, torch.zeros_like(bc1))

        psi_bc2_pred = model(train_bc2).squeeze()
        bc2 = psi_bc2_pred
        l4 = loss(bc2, torch.zeros_like(bc2))
        
        l1 /= X.shape[0]
        l2 /= X.shape[0]
        l3 /= train_bc1.shape[0]
        l4 /= train_bc2.shape[0]
        
        l = l1 + 1e-10*l2 + 1e-2*l3 + l4
        
        l.backward()
        mse += l.item()
        mse_1 += l1.item()
        mse_2 += l2.item()
        mse_3 += l3.item()
        mse_4 += l4.item()

        optimizer.step()

    mse /= n_batches
    mse_1 /= n_batches
    mse_2 /= n_batches
    mse_3 /= n_batches
    mse_4 /= n_batches

    epoch_scheduler.step()

    if min_loss > mse**0.5:
        min_loss = mse**0.5
        torch.save(model.state_dict(), "model.pt")
    loss_tot.append(mse)
    loss_1.append(mse_1)
    loss_2.append(mse_2)
    loss_3.append(mse_3)
    loss_4.append(mse_4)
    if i%10 == 0:
        print('Epoch: ' + str(i) + ' MSE: ' + str(mse) + 
              ' MSE 1: ' + str(mse_1) + ' MSE 2: ' + str(mse_2) + 
              ' MSE 3: ' + str(mse_3) + ' MSE 4: ' + str(mse_4))

fig = plt.figure(figsize=(12, 8))
fig.set_dpi(200)
plt.plot(np.log10(loss_tot))
plt.plot(np.log10(loss_1))
plt.plot(np.log10(loss_2))
plt.plot(np.log10(loss_3))
plt.plot(np.log10(loss_4))
plt.legend(['Total loss', 'MSE $\psi$', 'Loss Physics', 'Loss bc1', 'Loss bc2'], fontsize=15)
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Loss ($log_{10}$ scale)', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Training losses', fontsize=20)
