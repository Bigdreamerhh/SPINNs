import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt
from tqdm import tqdm

from tensorflow import keras
from tensorflow.keras import layers, initializers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects


gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

T = 8                         # The terminate time
c = 1                         # Wave speed
L = 2                         # The solution region is set as [-L, L]

ww = 2*L/10                   # The left end u is set as sin(2*pi*ww*t)
ff = 2*np.pi*ww

dx = 1e-2                     # Spatial mesh size
CFL = 0.9                     # The Courant-Friedrichs-Lewy number
dt = dx*CFL /c                # Time step

tt = np.arange(0,T,dt)
xx = np.arange(-L,L,dx)

nx = xx.shape[0]            # NoF for sapce
nt = tt.shape[0]            # NoF for time

u = np.zeros((nt, nx))

XX,TT = np.meshgrid(xx,tt)
Z = np.cos(TT)*np.cos(XX)

tx = np.concatenate((TT.reshape(-1,1), XX.reshape(-1,1)), axis = 1)

XX_flat = XX.reshape(-1,1)
TT_flat = TT.reshape(-1,1)


#%% Part 1: compare error
## ===============================================================================================

for i in range(2,nt):
    
    u[i, 0] = np.sin(ff*i*dt)      # h(t) = sin(2*pi*ww*t)
    for j in range(1,nx-1):
        u[i,j] = (CFL**2)*(u[i-1,j+1] + u[i-1, j-1]) + 2*(1-CFL**2)*u[i-1,j] - u[i-2, j]
   
    # Fixed boundary condtions
    u[i, -1] = 0



# norm_u = np.linalg.norm(u)

# for fig_path in ['fig', 'fig_new', 'fig_st']:

#     err_time = []

#     for i in range(70):
#         print(i)

#         u_dnn = tf.keras.models.load_model(fig_path+'/model_' + str(i))
#         u_pinn = u_dnn.predict(tx)
#         u_pinn = u_pinn.reshape(-1,XX.shape[1])

#         err = np.linalg.norm(u_pinn - u)/norm_u
#         err_time.append(err) 
    
#     np.savetxt(fig_path + '/err.txt', err_time)



err_time_P = np.loadtxt('fig/err.txt')
err_time_SP = np.loadtxt('fig_new/err.txt')
err_time_ST = np.loadtxt('fig_st/err.txt')

Ep_x = 25*np.arange(1,71)

plt.figure(1)
plt.plot(Ep_x, err_time_P, '-b', label = 'err_PINNs')
plt.plot(Ep_x, err_time_SP, '-r', label = 'err_SPINNs')
plt.plot(Ep_x, err_time_ST, '-g', label = 'err_Sobolev Training')
plt.xlabel('Epochs')
plt.ylabel('Relative Error')
plt.legend()
plt.savefig('err_compare.png')
plt.close()

#%% Part 2: compare solution
## ===============================================================================================

# fig, axs = plt.subplots(5, 3, sharex=True, sharey=True, layout="constrained", figsize=(10,10))

# path_list = ['fig', 'fig_st', 'fig_new']
# GAS_epoch_list = [10, 30, 50, 69]
# model_list = ['PINNs', 'Sobolev Training', 'SPINNs']

# for i in range(4):
#     print(i)
#     for j in range(3):
#         u_dnn = tf.keras.models.load_model(path_list[j]+'/model_' + str(GAS_epoch_list[i]))
#         u_pinn = u_dnn.predict(tx)
#         u_pinn = u_pinn.reshape(-1,XX.shape[1])
#         pcm = axs[i,j].pcolormesh(TT, XX, u_pinn, cmap='rainbow', vmin=-2, vmax=2)
#         axs[i,j].set(xlim=(0, 8), ylim=(-2, 2))
    
#     fig.colorbar(pcm, ax=axs[i,2])
#     axs[i,0].set_ylabel('$N_G=$'+str(2*i+1))
        
# for j in range(3):
#     u_dnn = tf.keras.models.load_model(path_list[j]+'/model_' + str(69))
#     u_pinn = u_dnn.predict(tx)
#     u_pinn = u_pinn.reshape(-1,XX.shape[1])
#     u_err = np.abs(u_pinn - u)
#     u_err = u_err.reshape(-1,XX.shape[1])
#     pcm = axs[4,j].pcolormesh(TT, XX, u_err, cmap='rainbow', vmin=0, vmax=1)
#     axs[4,j].set(xlim=(0, 8), ylim=(-2, 2))
#     axs[0,j].set_title(model_list[j])
    
# fig.colorbar(pcm, ax=axs[4,2])
# axs[4,0].set_ylabel('Error')
        
# plt.savefig('solution_compare.png')
# plt.close()


## =================== For Yuhui Liu ===============================================================

fig, axs = plt.subplots(3, 5, sharex=True, sharey=True, layout="constrained", figsize=(10,6))

path_list = ['fig', 'fig_st', 'fig_new']
GAS_epoch_list = [10, 30, 50, 69]
model_list = ['PINNs', 'Sobolev Training', 'SPINNs']

for i in range(3):
    print(i)
    for j in range(4):
        u_dnn = tf.keras.models.load_model(path_list[i]+'/model_' + str(GAS_epoch_list[j]))
        u_pinn = u_dnn.predict(tx)
        u_pinn = u_pinn.reshape(-1,XX.shape[1])
        pcm = axs[i,j].pcolormesh(TT, XX, u_pinn, cmap='rainbow', vmin=-2, vmax=2)
        axs[i,j].set(xlim=(0, 8), ylim=(-2, 2))
        axs[0,j].set_title('$N_G=$'+str(2*j+1))
    
    # fig.colorbar(pcm, ax=axs[i,2])
        
for j in range(3):
    u_dnn = tf.keras.models.load_model(path_list[j]+'/model_' + str(69))
    u_pinn = u_dnn.predict(tx)
    u_pinn = u_pinn.reshape(-1,XX.shape[1])
    u_err = np.abs(u_pinn - u)
    u_err = u_err.reshape(-1,XX.shape[1])
    pcm = axs[j,4].pcolormesh(TT, XX, u_err, cmap='rainbow', vmin=0, vmax=1)
    axs[j,4].set(xlim=(0, 8), ylim=(-2, 2))
    axs[j,0].set_ylabel(model_list[j])
    
# fig.colorbar(pcm, ax=axs[4,2])
axs[0,4].set_title('Error')
        
plt.savefig('solution_compare_yuhui.png')
plt.close()