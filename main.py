# This script is aimd at the testing of PINNs in the simulation of wave equations.
# We mainly focus on the issue of boundary condition and interface condition.

# Aim: Fixed BC or Free BC, denpends on the parameter Gamma 
# u_tt = u_xx ,  0 <= t <= T, -L <= x <= L
# with the initial and boundary condition u(0,x) = u_t(0,x) = 0, x(t,-L) = h(t), x(t,L) = 0

# By yuancheng@whu.edu.cn, 2023.5.23


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
#%% Part 0: Parameters
## ===============================================================================================

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

PINN_type = 1      # 0: classical   1: SPINNs
ST_mode = 1       # 0: classical   1: Sobolev training

ST_mode = ST_mode*PINN_type

if PINN_type >0:
    fig_path = 'fig_new'
else:
    fig_path = 'fig'

if ST_mode >0:
    fig_path = 'fig_st'



learning_rate = 1e-3
epoch_pinn = 250            # 500
epoch_gas = 7             # 10                        

bs_initial = 20          # 25
bs_left_bd = 20          # 25
bs_right_bd = 20         # 25
bs_inner = 1000             # 1000

num_Gaussian = 12          # 40
num_samples_perG = 50     # 200

k = 10          # train k*bs for the first epoch

total_step = 100000
# total_step = k*epochs

alpha = 1      # loss_weight for initial condition
beta = 1       # loss_weight for left boundary condition    
gamma = 1      # loss_weight for right boundary condition 

#%% Part 1: FDM method, Using finite difference method as a basisc referece solution
## ===============================================================================================

for i in range(2,nt):
    
    u[i, 0] = np.sin(ff*i*dt)      # h(t) = sin(2*pi*ww*t)
    for j in range(1,nx-1):
        u[i,j] = (CFL**2)*(u[i-1,j+1] + u[i-1, j-1]) + 2*(1-CFL**2)*u[i-1,j] - u[i-2, j]
   
    # Fixed boundary condtions
    u[i, -1] = 0
    
    # Absorbing boundary conditions
    # u[i,-1] = (1 - CFL/c)* u[i-1,-1] + CFL/c * u[i-1, -2]
    
    # if np.mod(i,3) == 1:
    #     f_xyz = open('fig/string_' + str(i) +'.xyz', 'w')
    #     f_xyz.writelines("  " + str(nx) + "\n")
    #     f_xyz.writelines("  wave solution\n")
    #     for j in range(nx-1):
    #         f_xyz.writelines("wave" + ' ' + str(j*dx)+ '   '+ str(u[i,j]) + ' ' + str(0) + '\n')
    #     f_xyz.writelines("wave" + ' ' + str((nx-1)*dx)+ '   '+ str(u[i,nx-1]) + ' ' + str(0))
    #     f_xyz.close()


# Visulization of the solution in t-x plane
XX,TT = np.meshgrid(xx,tt)
Z = np.cos(TT)*np.cos(XX)

tx = np.concatenate((TT.reshape(-1,1), XX.reshape(-1,1)), axis = 1)

XX_flat = XX.reshape(-1,1)
TT_flat = TT.reshape(-1,1)

norm_u = np.linalg.norm(u)

fig, ax = plt.subplots()
ax.pcolor(TT, XX, u, cmap='rainbow')
plt.savefig(fig_path + '/exact_solution.png')
plt.close()

#%% Part 2: PINNs 
## ===============================================================================================

class ResDNN_layer(keras.layers.Layer):
    def __init__(self, width = 40, act = 'relu', name = 'resdnn'):
        super(ResDNN_layer, self).__init__(name = name)
        self.initializer_W = tf.keras.initializers.RandomUniform(minval=-1e-1, maxval=1e-1)
        self.initializer = tf.keras.initializers.RandomUniform(minval=-1e-3, maxval=1e-3)
        self.H0 = layers.Dense(width, kernel_initializer= self.initializer_W, bias_initializer= self.initializer, activation = act, name = 'h0')
        # self.H1 = layers.Dense(width, kernel_initializer= self.initializer_W, bias_initializer= self.initializer, activation = act, name = 'h1')
        # self.H2 = layers.Dense(width, kernel_initializer= self.initializer_W, bias_initializer= self.initializer, activation = act, name = 'h2')
        # self.H3 = layers.Dense(width, kernel_initializer= self.initializer_W, bias_initializer= self.initializer, activation = act, name = 'h3')
        # self.res1 = layers.Dense(width, activation = act, name = 'res')
    def call(self, inputs):
        x0 = self.H0(inputs)
        # x1 = self.H1(x0)
        # x2 = self.H2(x1)
        # r1 = self.res1(inputs)
        # x = x0 + r1
        # x3 = self.H3(x)
        return x0

 
def build_model(act = 'relus', width = 40, depth = 3):
    model = keras.Sequential(name="my_sequential")
    initializer_W = tf.keras.initializers.RandomUniform(minval=-1, maxval=1.)
    initializer_b = tf.keras.initializers.RandomUniform(minval=-1e-0, maxval=1e-0)
    for i in range(depth):
        model.add(ResDNN_layer(width = width, act = act, name="res_layer_" + str(i+1)))
    model.add(layers.Dense(1, name = 'output', kernel_initializer = initializer_W, bias_initializer=initializer_b))
    return model


def relus(x):
    return tf.nn.relu(x**3)
    # return tf.nn.tanh(x)

get_custom_objects().update({'relus':Activation(relus)})

# u_dnn = DNN([50,100,100,100,100,80, 1], 'relus') 
# u_dnn = build_model(act = 'relus', width = 64, depth = 3)
u_dnn = tf.keras.models.load_model('ini_model') 

# u_pinn = u_dnn.predict(tx)
# u_pinn = u_pinn.reshape(-1,XX.shape[1])

# plt.figure(2)
# fig, ax = plt.subplots()
# ax.pcolor(TT, XX, u_pinn, cmap='rainbow')
# plt.savefig(fig_path + '/initial_pinn_solution.png')
# plt.close()

# u_dnn.save('ini_model')

# print(aaa)

#%% Part 3： training 
## ===============================================================================================

def InnerLoss(res,res_x, res_y):
    res = tf.cast(res, tf.float32)
    res_x = tf.cast(res_x, tf.float32)
    res_y = tf.cast(res_y, tf.float32)
    L = (res)**2 + ST_mode*((res_x)**2 + (res_y)**2)
    loss = K.mean(L)
    return loss


def InitialLoss(u,ut,ux):
    u = tf.cast(u, tf.float32)
    ut = tf.cast(ut, tf.float32)
    ux = tf.cast(ux, tf.float32)
    L = u**2 + ut**2 + 1 * PINN_type * ux**2
    loss = K.mean(L)
    return loss


def BouandaryLoss(u, g_data, ut,gt):
    u = tf.cast(u, tf.float32)
    g_data = tf.cast(g_data, tf.float32)
    ut = tf.cast(ut, tf.float32)
    gt = tf.cast(gt, tf.float32)
    L = (u - g_data)**2 + 1*PINN_type *(ut-gt)**2
    loss = K.mean(L)
    return loss


def RightLoss(u, ut):
    u = tf.cast(u, tf.float32)
    ut = tf.cast(ut, tf.float32)
    L = (u)**2 + 1*PINN_type * ut**2
    loss = K.mean(L)
    return loss


def find_local_max(tt,xx,rr, m = 10, epsilon = 0.2):
    # This section is left to be optimized
    local_max = []
    sort_index = np.argsort(-rr)
    local_max.append(sort_index[0])
    for i in range(rr.shape[0]):
        add_index = True
        for j in range(len(local_max)):
            r = (tt[sort_index[i]] - tt[local_max[j]])**2 + (xx[sort_index[i]] - xx[local_max[j]])**2
            if r < epsilon**2:
                add_index = False    
                break
        if add_index:
            local_max.append(sort_index[i])    
        if len(local_max)>m:
            break
    return local_max


def GM(centers, sigma= 0.1, G = 30):
    X_samples = centers
    for i in range(G):
        diff = sigma*np.random.normal(size=centers.shape)
        tmp = centers + diff
        X_samples = np.concatenate((X_samples, tmp), axis = 0)
    X_samples[X_samples[:,0]<0,0] = 0.01
    X_samples[X_samples[:,0]>T,0] = T-0.01
    X_samples[X_samples[:,1]<-L,1] = -L+0.01
    X_samples[X_samples[:,1]>L,1] = L-0.01
    return X_samples


def sampling(k, bs_initial =  bs_initial, bs_left_bd = bs_left_bd, bs_inner = bs_inner, bs_right_bd = bs_right_bd):
    # generate the samples in (t,x) space
    initial_data = -L + 2*L*np.random.rand(bs_initial*k, 2)
    initial_data[:,0] = 0

    leftbd_data = T*np.random.rand(bs_left_bd*k,2)
    leftbd_data[:,1] = -L
    
    rightbd_data = T*np.random.rand(bs_right_bd*k,2)
    rightbd_data[:,1] = L
    
    h_data = np.zeros(bs_left_bd*k)
    h_data = np.sin(ff*leftbd_data[:,0])
    ht_data = ff*np.cos(ff*leftbd_data[:,0])
    
    inner_data = np.zeros((bs_inner*k,2))
    inner_data[:,0] = T * np.random.rand(bs_inner*k)
    inner_data[:,1] = -L + 2*L*np.random.rand(bs_inner*k)
    return initial_data, leftbd_data, np.float32(h_data), np.float32(ht_data), inner_data, rightbd_data


initial_learning_rate = learning_rate
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=total_step,
    decay_rate=1,
    staircase=True)


optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)


@tf.function
def train_step(initial_data, left_bd_data, h_data, ht_data, inner_data, rbd_data):

    bs_initial = initial_data.shape[0]
    bs_left_bd = left_bd_data.shape[0]
    bs_inner = inner_data.shape[0]
    
    in_bd_data = tf.concat([initial_data, left_bd_data, inner_data, rbd_data], axis = 0)
    
    with tf.GradientTape() as tape4:
        
        with tf.GradientTape() as tape3:
            tape3.watch(in_bd_data)
        
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(in_bd_data)
        
                with tf.GradientTape() as tape2:
                    tape2.watch(in_bd_data)
                    u = u_dnn(in_bd_data)
        
                du = tape2.gradient(u, in_bd_data)
                c = tf.stack([du[:,i:i+1] for i in range(2)], axis = -1)
                d = tf.concat([tape1.gradient(c[:,:,i], in_bd_data) for i in range(2)], axis = -1)
            
            Hess = tf.reshape(d, [-1, 2, 2])
            ddu = tf.linalg.diag_part(Hess)
            # inner term (residual)
            residual = ddu[:,0] - ddu[:,1]
        
        d_residual = tape3.gradient(residual, in_bd_data)
        residual_x = d_residual[:,0]
        residual_y = d_residual[:,1]
        res_x = residual_x[bs_initial + bs_left_bd:bs_initial+bs_left_bd+bs_inner]       
        res_y = residual_y[bs_initial + bs_left_bd:bs_initial+bs_left_bd+bs_inner]

        res = residual[bs_initial + bs_left_bd:bs_initial+bs_left_bd+bs_inner]

        # right boundary (test for ABC, FixBC, FreeBC)
        abc = du[:,0] + du[:,1]
        res_abc = abc[bs_initial + bs_left_bd + bs_inner:]
        u_rbd = u[bs_initial + bs_left_bd + bs_inner:]
        ut_rbd = du[bs_initial + bs_left_bd + bs_inner:,0]

        # initial term
        u_ini = u[:bs_initial]
        ut_ini = du[:bs_initial,0]
        ux_ini = du[:bs_initial,1]

        # left boundary term
        u_bd = u[bs_initial : bs_initial + bs_left_bd]
        ut_bd = du[bs_initial : bs_initial + bs_left_bd, 0]

        loss_1 = InitialLoss(u_ini, ut_ini, ux_ini)
        loss_2 = BouandaryLoss(u_bd[:,0], h_data, ut_bd, ht_data)
        loss_3 = InnerLoss(res, res_x, res_y)
        loss_4 = RightLoss(u_rbd, ut_rbd)

        loss = (alpha*loss_1 + beta*loss_2 + loss_3 + gamma * loss_4)

        loss_abc = K.mean(K.abs(res_abc))

    grads = tape4.gradient(loss, u_dnn.trainable_weights)    
    optimizer.apply_gradients(zip(grads, u_dnn.trainable_weights))
    return loss, grads, loss_abc

@tf.function
def cal_residual(inner_data):
    with tf.GradientTape() as tape3:
        in_bd_data = tf.concat([inner_data], axis = 0)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(in_bd_data)
            with tf.GradientTape() as tape2:
                tape2.watch(in_bd_data)
                u = u_dnn(in_bd_data)
            du = tape2.gradient(u, in_bd_data)
            c = tf.stack([du[:,i:i+1] for i in range(2)], axis = -1)
            d = tf.concat([tape1.gradient(c[:,:,i], in_bd_data) for i in range(2)], axis = -1)
        
        Hess = tf.reshape(d, [-1, 2, 2])
        ddu = tf.linalg.diag_part(Hess)
        residual = ddu[:,0] - ddu[:,1]
        
    return residual
    # return Hess, ddu

loss_time = []
abc_time = []
err_time = []

##################################################################################################################
###################################### Main Loop #################################################################
##################################################################################################################

initial_data, leftbd_data, hh_data, hht_data, inner_data, righbd_data = sampling(k)

kk = 0

for gas_step in range(epoch_gas):

    if gas_step > 0:
        rr = cal_residual(tx)
        rr = rr.numpy()
        RR = rr.reshape(-1, XX.shape[1])
        
        # top_res_index = np.argsort(residual)[:30]
        top_res_index = find_local_max(TT_flat, XX_flat, rr**2, m = num_Gaussian)
        
        center_t = TT_flat[top_res_index]
        center_x = XX_flat[top_res_index]

        centers = np.concatenate((center_t, center_x), axis = 1)
        add_samples = GM(centers, G = num_samples_perG)

        inner_data = np.concatenate((inner_data, add_samples), axis = 0)

        # adding 

        num_add_initial = int(add_samples.shape[0] * bs_initial/bs_inner)
        num_add_bd = int(add_samples.shape[0]*bs_left_bd/bs_inner)

        add_initial_data = -L + 2*L*np.random.rand(num_add_initial, 2)
        add_initial_data[:,0] = 0

        add_leftbd_data = T*np.random.rand(num_add_bd,2)
        add_leftbd_data[:,1] = -L

        add_rightbd_data = T*np.random.rand(num_add_bd,2)
        add_rightbd_data[:,1] = L

        add_hh_data = np.zeros(num_add_bd)
        add_hh_data = np.sin(ff*add_leftbd_data[:,0])

        add_hht_data = np.zeros(num_add_bd)
        add_hht_data = ff*np.cos(ff*add_leftbd_data[:,0])


        initial_data = np.concatenate((initial_data, add_initial_data), axis = 0)
        leftbd_data = np.concatenate((leftbd_data, add_leftbd_data), axis = 0)
        hh_data = np.concatenate((hh_data, add_hh_data), axis = 0)
        hht_data = np.concatenate((hht_data, add_hht_data), axis = 0)

        righbd_data = np.concatenate((righbd_data, add_rightbd_data), axis = 0)
        
        # plt.figure(1)
        # fig, ax = plt.subplots()
        # ax.pcolor(TT, XX, RR, cmap='rainbow')
        # plt.plot(add_samples[:,0], add_samples[:,1], '+', markersize = 0.5, c='red')
        # # plt.plot(inner_data[:,0], inner_data[:,1], '.', markersize = 0.1, c='black')
        # plt.savefig(fig_path + '/residual_with_gas_' + str(gas_step) + '.png')
        # plt.close()
        
        u_pinn = u_dnn.predict(tx)
        u_pinn = u_pinn.reshape(-1,XX.shape[1])

        plt.figure(2)
        fig, ax = plt.subplots()
        ax.pcolor(TT, XX, u_pinn, cmap='rainbow')
        plt.savefig(fig_path + '/pinn_solution_' + str(gas_step) + '.png')
        plt.close()
        
        # plt.figure(3)
        # fig, ax = plt.subplots()
        # plt.plot(inner_data[:,0], inner_data[:,1], '+', markersize = 0.5, c='blue')
        # plt.savefig(fig_path + '/samples_' + str(gas_step) + '.png')
        # plt.close()

    for epoch in range(epoch_pinn):
        
        dataset_ini = tf.data.Dataset.from_tensor_slices(initial_data)
        dataset_bd = tf.data.Dataset.from_tensor_slices((leftbd_data, hh_data, hht_data))
        dataset_in = tf.data.Dataset.from_tensor_slices(inner_data)
        dataset_right = tf.data.Dataset.from_tensor_slices(righbd_data)

        train_dataset_ini = dataset_ini.shuffle(40000).batch(bs_initial)
        train_dataset_bd = dataset_bd.shuffle(40000).batch(bs_left_bd)
        train_dataset_in = dataset_in.shuffle(40000).batch(bs_inner)
        train_dataset_rbd = dataset_right.shuffle(40000).batch(bs_right_bd)

        train_list_ini = list(train_dataset_ini.as_numpy_iterator())
        train_list_bd = list(train_dataset_bd.as_numpy_iterator())
        train_list_in = list(train_dataset_in.as_numpy_iterator())
        train_list_rbd = list(train_dataset_rbd.as_numpy_iterator())

        num_step = min(len(train_list_in), len(train_list_bd), len(train_list_ini), len(train_list_rbd))

        if epoch%25 == 0:
            
            u_dnn.save(fig_path +'/model_'+str(kk))
            kk += 1
            
            # u_pinn = u_dnn.predict(tx)
            # u_pinn = u_pinn.reshape(-1,XX.shape[1])

            # err = np.linalg.norm(u_pinn - u)/norm_u
            # err_time.append(err)

        print('='*60)

        for step in range(num_step):

            ini_data = train_list_ini[step]
            (bd_data, h_data, ht_data) = train_list_bd[step]
            in_data = train_list_in[step]
            rbd_data = train_list_rbd[step]

            # if (epoch<1) & (step < 1):
                # loss = cal_loss(ini_data, bd_data, h_data, in_data)
                # loss_0 = loss.numpy()
                # loss_0 = 0
                # loss_time.append(loss_0)
                # print("Initial loss = ", loss_0)

            loss, grads, loss_abc = train_step(ini_data, bd_data, h_data, ht_data, in_data, rbd_data)

            if step%4 ==0:
    
                f_log = open('log.txt', 'a')
                f_log.writelines("Epoch_gas = %d, Epoch_pinn = %d, step = %d, loss = %.7f, loss_abc = %.7f \n"
                   % (gas_step, epoch, step, loss.numpy(), loss_abc.numpy()))
                f_log.close()
    
                print("Epoch_gas = %d, Epoch_pinn = %d, step = %d, loss = %.7f, loss_abc = %.7f \n"
                   % (gas_step, epoch, step, loss.numpy(), loss_abc.numpy()))
    
        loss_time.append(loss.numpy())
        abc_time.append(loss_abc.numpy())    


final_loss = np.mean(loss_time[-10:])
u_dnn.save('model')

loss_time
# plot the loss and the prediction.

np.savetxt(fig_path + '/loss.txt', loss_time)
np.savetxt(fig_path + '/abc.txt', abc_time)
np.savetxt(fig_path + '/err.txt', err_time)

plt.figure(1)
plt.plot(loss_time, 'b', label = 'loss')
plt.plot(abc_time, 'r', label = 'abc')
plt.legend()
plt.savefig(fig_path + '/loss.png')
plt.close()


plt.figure(2)
plt.plot(err_time, 'g', label = 'err')
plt.title('Three Lines')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.savefig(fig_path + '/err.png')
plt.close()


u_pinn = u_dnn.predict(tx)
u_pinn = u_pinn.reshape(-1,XX.shape[1])
plt.figure(3)
fig, ax = plt.subplots()
ax.pcolor(TT, XX, u_pinn, cmap='rainbow')
plt.savefig(fig_path + '/pinn_solution.png')
plt.close()

# err_time = np.loadtxt('fig/err.txt')
# err_time0 = np.loadtxt('fig/err0.txt')
# err_time3 = np.loadtxt('fig/err3.txt')

# plt.figure(1)
# plt.plot(err_time, 'b', label = 'err1')
# plt.plot(err_time0, 'r', label = 'err0')
# plt.plot(err_time3, 'g', label = 'err3')
# plt.legend()
# plt.savefig(fig_path + '/err_compare.png')
# plt.close()