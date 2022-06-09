import math
import numpy as np
import pandas as pd
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import matplotlib as mpl


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
t_si = 5e-7
epsilon_0 = 8.85418781e-12
epsilon_si = epsilon_0*11.9
epsilon_sio2 = epsilon_0*3.9
delta_psi_MS = 0.21
psi_t = 26e-3
n_i = 1e16
psi_F = psi_t*math.log(N_A/n_i)
q = 1.6e-19

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

y = np.concatenate([np.linspace(0, 5e-8, 100, endpoint=False), np.linspace(5e-8, 4e-7, 100)])
psi = np.zeros((2, y.size))

fig, ax = plt.subplots(figsize=(12, 10))
fig.set_dpi(200)
Vgs = np.concatenate([np.linspace(-2, 0, 100, endpoint=False), np.linspace(0, 1, 50, endpoint=False), np.linspace(1, 2, 100)])

psi_out = []
for i in Vgs:
    Vg = i
    sol = solve_bvp(fun, bc, y, psi, tol=1e-3, max_nodes=20000)
    plt.plot(y, sol.sol(y)[0], label='$V_{gs}$=' + '%.2f'%i + ' V')
    psi_out.append(list(sol.sol(y)[0]))
# plt.legend(loc='center right', fontsize=15, bbox_to_anchor=(1.25, 0.5))
plt.xlabel("Vertical Distance, y (nm)", fontsize=20)
plt.ylabel("Potential, $\psi(y)$    (V)", fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
ticks = ax.get_xticks()[1:]*10**9
ax.set_xticklabels(ticks)
plt.xlim(0)
plt.show()

psi_out = np.array(psi_out)
psi_out = psi_out.flatten()
Vgs_1 = []
for i in Vgs:
    Vgs_1 += [i]*len(list(y))


df = {'y':list(y)*len(list(Vgs)), 'Vgs':Vgs_1, 'psi': list(psi_out)}
df = pd.DataFrame(df)

df_b_0 = df[df['y'] == 0]
df_b_1 = df[df['y'] == 4e-7]


df.to_csv('../Data/data_y_100_100_58_vgs_100_50_100.csv', header=True, index=False)
df_b_0.to_csv('../Data/data_bc1_vgs_100_50_100.csv', header=True, index=False)
df_b_1.to_csv('../Data/data_bc2_vgs_100_50_100.csv', header=True, index=False)

yy = np.linspace(0, 4e-7, 500)
psii = np.zeros((2, yy.size))
Vgs = np.linspace(-5, 5, 50)
psi_zero = []
psi_zero_pred = []
for i in Vgs:
    Vg = i
    sol = solve_bvp(fun, bc, yy, psii, tol=1e-3, max_nodes=20000)
    psi_zero.append(sol.sol(yy)[0][0])

fig, ax = plt.subplots(figsize=(12, 8))
fig.set_dpi(200)
ax.axvline(-0.6, ls='--', lw=1, c='#000000')
ax.axvline(1.2, ls='--', lw=1, c='#000000')
fig.text(0.65,0.2,'Strong Inversion', color = 'black', size=15)
fig.text(0.48,0.2,'Depletion', color = 'black', size=15)
fig.text(0.25,0.2,'Accumulation', color = 'black', size=15)
plt.plot(Vgs, psi_zero)
plt.xlabel("Gate Voltage, $V_{gs}$  (V)", fontsize=20)
plt.ylabel("Surface potential, $\psi(0)$", fontsize=20)
plt.xticks([i for i in range(-5, 6)], fontsize=15)
plt.yticks(fontsize=15)
plt.show()
