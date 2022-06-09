import math
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from sklearn.metrics import mean_squared_error

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

path = '../Data/data_y_100_100_58_vgs_100_50_100.csv'
data_f = pd.read_csv(path)

X = data_f.drop('psi', axis = 1).values.astype(np.float32)
train = torch.tensor(X).to(device)
mean = torch.mean(train, axis=0)
std = torch.std(train, axis=0)

class NN(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_layers):
        super().__init__()
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
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

model = NN(2, 1, 20, 4).to(device)
model.load_state_dict(torch.load('../Trained Models/Standard PINN/model.pt', map_location='cuda:0'))

def fun(y, psi):
    A = q*N_A/epsilon_si
    first = psi[1]
    second = -A*(np.exp(-psi[0]/psi_t) - 1 - np.exp(-2*psi_F/psi_t)*(np.exp(psi[0]/psi_t) - 1))
    return np.array([first, second])

def bc(psi_a, psi_b):
    Cox = epsilon_sio2/t_ox
    B = Cox/epsilon_si
    first = +psi_a[1] + B*(Vg - psi_a[0])
    second = psi_b[0]
    return np.array([first, second])

yy = np.linspace(0, 4e-7, 200)
psii = np.zeros((2, yy.size))
fig = plt.figure(figsize=(10, 8))
fig.set_dpi(200)
Vgs = np.linspace(-2, 2, 21)
color = iter(plt.cm.rainbow(np.linspace(-1, 1.5, 21)))
for i in Vgs:
    Vg = i
    c = next(color)
    sol = solve_bvp(fun, bc, yy, psii, tol=1e-3, max_nodes=20000)
    plt.plot(yy, sol.sol(yy)[0], 'o', c=c, markevery=10, mfc='none')
    vgs = np.ones_like(yy)*i
    inp = np.vstack([yy, vgs])
    inp = np.transpose(inp).astype(np.float32)
    inp = torch.tensor(inp).to(device)
    psi = model(inp)
    psi = psi.cpu().detach().numpy()
    plt.plot(yy, psi, c=c, label='$V_{gs}$=' + '%.2f'%i + ' V')
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
plt.xlabel("Vertical Distance, $y$ (nm)", fontsize=20)
plt.ylabel("Potential, $\psi(y)$    (V)", fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlim(0)
plt.show()


yy = np.linspace(0, 4e-7, 200)
psii = np.zeros((2, yy.size))
fig = plt.figure(figsize=(10, 8))
fig.set_dpi(200)
Vgs = np.linspace(-5, -2, 21)
color = iter(plt.cm.rainbow(np.linspace(-1, 1.5, 21)))
for i in Vgs:
    Vg = i
    c = next(color)
    sol = solve_bvp(fun, bc, yy, psii, tol=1e-3, max_nodes=20000)
    plt.plot(yy, sol.sol(yy)[0], 'o', c=c, markevery=10, mfc='none')
    vgs = np.ones_like(yy)*i
    inp = np.vstack([yy, vgs])
    inp = np.transpose(inp).astype(np.float32)
    inp = torch.tensor(inp).to(device)
    psi = model(inp)
    psi = psi.cpu().detach().numpy()
    plt.plot(yy, psi, c=c, label='$V_{gs}$=' + '%.2f'%i + ' V')
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
plt.xlabel("Vertical Distance, $y$ (nm)", fontsize=20)
plt.ylabel("Potential, $\psi(y)$    (V)", fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlim(0)
plt.show()


yy = np.linspace(0, 4e-7, 200)
psii = np.zeros((2, yy.size))
fig = plt.figure(figsize=(10, 8))
fig.set_dpi(200)
Vgs = np.linspace(2, 5, 21)
color = iter(plt.cm.rainbow(np.linspace(-1, 1.5, 21)))
for i in Vgs:
    Vg = i
    c = next(color)
    sol = solve_bvp(fun, bc, yy, psii, tol=1e-3, max_nodes=20000)
    plt.plot(yy, sol.sol(yy)[0], 'o', c=c, markevery=10, mfc='none')
    vgs = np.ones_like(yy)*i
    inp = np.vstack([yy, vgs])
    inp = np.transpose(inp).astype(np.float32)
    inp = torch.tensor(inp).to(device)
    psi = model(inp)
    psi = psi.cpu().detach().numpy()
    plt.plot(yy, psi, c=c, label='$V_{gs}$=' + '%.2f'%i + ' V')
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
plt.xlabel("Vertical Distance, $y$ (nm)", fontsize=20)
plt.ylabel("Potential, $\psi(y)$    (V)", fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlim(0)
plt.show()

psii = np.zeros((2, yy.size))
Vgs = np.linspace(-5, 5, 201)
psi_zero = []
psi_zero_pred = []
for i in Vgs:
    Vg = i
    sol = solve_bvp(fun, bc, yy, psii, tol=1e-3, max_nodes=20000)
    psi_zero.append(sol.sol(yy)[0][0])

    zeros = np.zeros_like(yy)
    vgs = np.ones_like(yy)*i
    inp = np.vstack([zeros, vgs])
    inp = np.transpose(inp).astype(np.float32)
    inp = torch.tensor(inp).to(device)
    psi = model(inp)
    psi = psi.cpu().detach().numpy()
    psi_zero_pred.append(psi[0])

fig = plt.figure()
fig.set_dpi(200)
plt.plot(Vgs, psi_zero, 'o', markevery=10, mfc='none')
plt.plot(Vgs, psi_zero_pred)
plt.legend(['Ground Truth', 'Predicted'])
plt.xlabel("Gate Voltage, $V_{gs}$  (V)", fontsize=20)
plt.ylabel("Surface potential, $\psi(0)$", fontsize=20)
plt.xticks([i for i in range(-5, 6)], fontsize=15)
plt.yticks(fontsize=15)
plt.show()

yy = np.linspace(0, 4e-7, 1000)
psii = np.zeros((2, yy.size))
mse = []
mse_phy = []
Vgs = np.linspace(-5, 5, 500)
for i in Vgs:
    Vg = i
    sol = solve_bvp(fun, bc, yy, psii, tol=1e-3, max_nodes=20000)
    vgs = np.ones_like(yy)*i
    inp = np.vstack([yy, vgs])
    inp = np.transpose(inp).astype(np.float32)
    inp = torch.tensor(inp, requires_grad=True).to(device)
    psi = model(inp)
    d_psi = torch.autograd.grad(psi, inp, torch.ones_like(psi), create_graph=True)[0]
    d_psi_2 = torch.autograd.grad(d_psi, inp, torch.ones_like(d_psi), create_graph=True)[0]
    d_psi_2 = d_psi_2[:, 0].unsqueeze(dim=1)
    f = d_psi_2/ A +  (torch.exp(-psi/psi_t) - 1 - C*(torch.exp(psi/psi_t) - 1))

    mse_phy.append(float(torch.mean(f**2).cpu().detach().numpy()))
    psi = psi.cpu().detach().numpy()
    mse.append(mean_squared_error(list(sol.sol(yy)[0]), psi))
np.mean(mse), np.mean(mse_phy)

import warnings
warnings.filterwarnings("ignore")
yy = np.linspace(0, 4e-7, 2)
psii = np.zeros((2, yy.size))
mse_bc1 = []
mse_bc2 = []
Vgs = np.linspace(-5, 5, 1001)
for i in Vgs:
    Vg = i
    sol = solve_bvp(fun, bc, yy, psii, tol=1e-3, max_nodes=20000)
    vgs = np.ones_like(yy)*i
    inp = np.vstack([yy, vgs])
    inp = np.transpose(inp).astype(np.float32)
    inp = torch.tensor(inp, requires_grad=True).to(device)
    psi = model(inp)
    d_psi_bc1 = torch.autograd.grad(psi, inp, torch.ones_like(psi), create_graph=True)[0]
    d_psi_bc1 = d_psi_bc1[:, 0][0]
    bc1 = d_psi_bc1 / B + (inp[:, 1][0] - psi[0])

    mse_bc1.append(float(torch.mean(bc1**2).cpu().detach().numpy()))
    mse_bc2.append(float(torch.mean(psi[1]**2).cpu().detach().numpy()))
    psi = psi.cpu().detach().numpy()
np.mean(mse_bc1), np.mean(mse_bc2)

yy = np.linspace(0, 4e-7, 100)
psii = np.zeros((2, yy.size))
Vgs = np.linspace(-5, 5, 1001)
mse = []
for i in Vgs:
    Vg = i
    sol = solve_bvp(fun, bc, yy, psii, tol=1e-3, max_nodes=20000)
    zeros = np.zeros_like(yy)
    vgs = np.ones_like(yy)*i
    inp = np.vstack([zeros, vgs])
    inp = np.transpose(inp).astype(np.float32)
    inp = torch.tensor(inp).to(device)
    psi = model(inp)
    psi = psi.cpu().detach().numpy()
    
    mse.append((sol.sol(yy)[0][0] - psi[0])**2)
np.mean(mse)

