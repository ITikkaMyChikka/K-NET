###############
### IMPORTS ###
###############
# Generic Imports
from datetime import datetime
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np
import os
# KalmanNet_TSP imports
from src.KalmanNet_TSP import Linear_sysmdl
from src.KalmanNet_TSP import KF_Pipeline
from filterpy.kalman import KalmanFilter
# My imports
import src.Linear_models.CA2D_mm as CA
from src.data_gen.linear_data_generator import DataGen
import src.parameters as params

############
### CUDA ###
############

if torch.cuda.is_available():
    cuda0 = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("Running on the GPU")
else:
    cuda0 = torch.device("cpu")
    print("Running on the CPU")

################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)

#############
### PATHS ###
#############
# Paths to folders
results_path = "/Users/sidhu/Documents/ETH/Semester Project/Adria_KN/src/Sim1/Results/"
plots_path = "/Users/sidhu/Documents/ETH/Semester Project/K-NET/Results/"
DatafolderName = 'SyntheticData/EKFNet_paper/'

#######################################
### DEFINE MOTION/STATE SPACE MODEL ###
#######################################

# Data Generation
F_gen = CA.F_gen
Q_gen = CA.Q_gen
H_gen = CA.H_gen
R_gen = CA.R_gen

# Full Observation
F_FO = CA.F_FO
Q_FO = CA.Q_FO
H_FO = CA.H_FO
R_FO = CA.R_FO

# Position Observation Model
F_PO = CA.F_PO
Q_PO = CA.Q_PO
H_PO = CA.H_PO
R_PO = CA.R_PO

# Acceleration Observation Model
F_AO = CA.F_AO
Q_AO = CA.Q_AO
H_AO = CA.H_AO
R_AO = CA.R_AO

# Position and Acceleration Model
F_PAO = CA.F_PAO
Q_PAO = CA.Q_PAO
H_PAO = CA.H_PAO
R_PAO = CA.R_PAO


# Define Trajectory Length
T = 300000  # T-train defines the trajectory length for training&validation dataset
T_decimated = int(T / params.ratio_ca)

# Initialize SystemModel for Data Generation
sys_model_gen = Linear_sysmdl.SystemModel(F=F_gen, Q=Q_gen, H=H_gen, R=R_gen, T=T)

# Initialize initial values for state vectors and covariance matrix
# Initial velocity corresponds to 18.0556m/s = 65km/h, then we start to either accelerate or decelerate
m1x_0 = torch.tensor([0, 2.0556, 0, 0, 2.0556, 0]).float()  # x_0 = [px, vx, ax, py, vy, ay]
m2x_0 = 0.0 * torch.eye(6, 6).float()
sys_model_gen.InitSequence(m1x_0=m1x_0, m2x_0=m2x_0)

###############################
### GENERATE SYNTHETIC DATA ###
###############################
dataFileName = ['CA_synthetic']  # Naming the dataset
"""
# We only want to generate data if the folder is empty
#if os.path.isfile("./" + dataFileName[2]):
print("GENERATING DATA")
# Generating data
DataGen(sys_model_gen, DatafolderName + dataFileName[0], T)

##############################
### LOADING SYNTHETIC DATA ###
##############################
print("Data Load")
[input, target] = torch.load(DatafolderName + dataFileName[0], map_location=cuda0)

print("testset size:", target.size())
print("testset input size:", input.size())

#######################
### DECIMATING DATA ###
#######################
print("DECIMATING DATA")
target_shape = target.shape
input_shape = input.shape
ratio = params.ratio_ca
target_dec = torch.zeros((target_shape[0], target_shape[1], int(target_shape[2]/ratio)))
input_dec = torch.zeros((input_shape[0], input_shape[1], int(input_shape[2] / ratio)))
for sample in range(0, input_shape[0]):
    for t in range(0, int(input_shape[2]/ratio)):  # We make 100'000/1000 step (#KF steps)
        target_dec[sample, :, t] = target[sample, :, t*ratio]
        input_dec[sample, :, t] = input[sample, :, t*ratio]



####################
### ADDING NOISE ###
####################
print("PERTURBING DATA")
perturbation = params.perturbation_cv
target_shape = target_dec.shape
input_shape = input_dec.shape
target_per = torch.zeros_like(target_dec)
input_per = torch.zeros_like(input_dec)
for sample in range(0, target_shape[0]):
    for t in range(0, target_shape[2]):
        mean = torch.zeros(R_FO.size()[0])  # We use the Full Observation R Matrix to perturb the data and obtain the observation
        er = np.random.multivariate_normal(mean, R_FO, 1)
        input_per[sample, :, t] = input_dec[sample, :, t] + er
        target_per[sample, :, t] = target_dec[sample, :, t]

        #noise_target = torch.randn(target_shape[1])
        #noise_input = torch.randn(input_shape[1])
        #target_per[sample, :, t] = target_dec[sample, :, t] + noise_target * perturbation
        #input_per[sample, :, t] = input_dec[sample, :, t] + noise_input * perturbation


##########################################################
### SAVING THE GT DATA, DECIMATED DATA, PERTURBED DATA ###
##########################################################
torch.save([input, target, target_dec, input_dec, target_per, input_per], DatafolderName + dataFileName[0])
"""

####################
### LOADING DATA ###
####################
print("LOADING FINAL DATA")
print(dataFileName[0])
[input, target, target_dec, input_dec, target_per, input_per] = torch.load(DatafolderName + dataFileName[0], map_location=cuda0)


############################
### POS ONLY OBSERVATION ###
############################

input_PO = input_per[:, 0:6:3, :]


###############################
### POS AND ACC OBSERVATION ###
###############################

input_PAO = torch.cat((input_per[:, 0:3:2, :], input_per[:, 3:6:2, :]), 1)


#######################
### ACC OBSERVATION ###
#######################
input_AO = input_per[:, 2:6:3, :]




####################
### KALMANFILTER ###
####################
print("KALMAN FILTERS RUNNING")

# Initialize empty files to save results of KF
FO_MSE_avg_best, FO_MSE_pos_best, FO_MSE_vel_best, FO_MSE_acc_best, FO_KF_best, q_FO_best = None, None, None, None, None, None
PO_MSE_avg_best, PO_MSE_pos_best, PO_MSE_vel_best, PO_MSE_acc_best, PO_KF_best, q_PO_best = None, None, None, None, None, None
AO_MSE_avg_best, AO_MSE_pos_best, AO_MSE_vel_best, AO_MSE_acc_best, AO_KF_best, q_AO_best = None, None, None, None, None, None
PAO_MSE_avg_best, PAO_MSE_pos_best, PAO_MSE_vel_best, PAO_MSE_acc_best, PAO_KF_best, q_PAO_best = None, None, None, None, None, None
"""
# System Model Initialization for different observation models
model_FO = Linear_sysmdl.SystemModel(F=F_FO, Q=Q_FO*params.q_ca*params.ratio_ca, H=H_FO, R=R_FO, T=T_decimated)
model_PO = Linear_sysmdl.SystemModel(F=F_PO, Q=Q_PO*params.q_ca*params.ratio_ca, H=H_PO, R=R_PO, T=T_decimated)
model_AO = Linear_sysmdl.SystemModel(F=F_AO, Q=Q_AO*params.q_ca*params.ratio_ca, H=H_AO, R=R_AO, T=T_decimated)
model_PAO = Linear_sysmdl.SystemModel(F=F_PAO, Q=Q_PAO*params.q_ca*params.ratio_ca, H=H_PAO, R=R_PAO, T=T_decimated)

# Initialize System Models with start conditions
model_FO.InitSequence(m1x_0=m1x_0, m2x_0=m2x_0)
model_PO.InitSequence(m1x_0=m1x_0, m2x_0=m2x_0)
model_AO.InitSequence(m1x_0=m1x_0, m2x_0=m2x_0)
model_PAO.InitSequence(m1x_0=m1x_0, m2x_0=m2x_0)

###########################
### KF without q_search ###
###########################
print("Testing KF with full observation")
[FO_MSE_avg, FO_MSE_pos, FO_MSE_vel, FO_MSE_acc, FO_KF] = KF_Pipeline.KF_Test1(model_FO, input_per, target_per)
FO_MSE_avg_best, FO_MSE_pos_best, FO_MSE_vel_best, FO_MSE_acc_best, FO_KF_best, q_FO_best = FO_MSE_avg, FO_MSE_pos, FO_MSE_vel, FO_MSE_acc, FO_KF, params.q_ca

print("Testing KF with position observation")
[PO_MSE_avg, PO_MSE_pos, PO_MSE_vel, PO_MSE_acc, PO_KF] = KF_Pipeline.KF_Test1(model_PO, input_PO, target_per)
PO_MSE_avg_best, PO_MSE_pos_best, PO_MSE_vel_best, PO_MSE_acc_best, PO_KF_best, q_PO_best = PO_MSE_avg, PO_MSE_pos, PO_MSE_vel, PO_MSE_acc, PO_KF, params.q_ca

print("Testing KF with acceleration observation")
[AO_MSE_avg, AO_MSE_pos, AO_MSE_vel, AO_MSE_acc, AO_KF] = KF_Pipeline.KF_Test1(model_AO, input_AO, target_per)
AO_MSE_avg_best, AO_MSE_pos_best, AO_MSE_vel_best, AO_MSE_acc_best, AO_KF_best, q_AO_best = AO_MSE_avg, AO_MSE_pos, AO_MSE_vel, AO_MSE_acc, AO_KF, params.q_ca

print("Testing KF with position and acceleration observation")
[PAO_MSE_avg, PAO_MSE_pos, PAO_MSE_vel, PAO_MSE_acc, PAO_KF] = KF_Pipeline.KF_Test1(model_PAO, input_PAO, target_per)
PAO_MSE_avg_best, PAO_MSE_pos_best, PAO_MSE_vel_best, PAO_MSE_acc_best, PAO_KF_best, q_PAO_best = PAO_MSE_avg, PAO_MSE_pos, PAO_MSE_vel, PAO_MSE_acc, PAO_KF, params.q_ca
"""

########################
### KF WITH Q SEARCH ###
########################

q_values_FO = torch.arange(0, 0.001, 0.000001)
q_values_PO = torch.arange(0, 0.001, 0.000001)
q_values_AO = torch.arange(0, 0.001, 0.000001)
q_values_PAO = torch.arange(0, 0.001, 0.000001)


for q in q_values_FO:
    # System Model Initialization for different observation models
    model_FO = Linear_sysmdl.SystemModel(F=F_FO, Q=Q_FO * q * params.ratio_ca, H=H_FO, R=R_FO, T=T_decimated)
    # Initialize System Models with start conditions
    model_FO.InitSequence(m1x_0=m1x_0, m2x_0=m2x_0)

    [FO_MSE_avg, FO_MSE_pos, FO_MSE_vel, FO_MSE_acc, FO_KF] = KF_Pipeline.KF_Test1(model_FO, input_per, target_per)

    if FO_MSE_avg_best is None or FO_MSE_avg_best > FO_MSE_avg:
        FO_MSE_avg_best, FO_MSE_pos_best, FO_MSE_vel_best, FO_MSE_acc_best, FO_KF_best, q_FO_best = FO_MSE_avg, FO_MSE_pos, FO_MSE_vel, FO_MSE_acc, FO_KF, q

for q in q_values_PO:
    # System Model Initialization For different observation models
    model_PO = Linear_sysmdl.SystemModel(F=F_PO, Q=Q_PO * q * params.ratio_ca, H=H_PO, R=R_PO, T=T_decimated)
    # Initialize System Models with start conditions
    model_PO.InitSequence(m1x_0=m1x_0, m2x_0=m2x_0)

    [PO_MSE_avg, PO_MSE_pos, PO_MSE_vel, PO_MSE_acc, PO_KF] = KF_Pipeline.KF_Test1(model_PO, input_PO, target_per)

    if PO_MSE_avg_best is None or PO_MSE_avg_best > PO_MSE_avg:
        PO_MSE_avg_best, PO_MSE_pos_best, PO_MSE_vel_best, PO_MSE_acc_best, PO_KF_best, q_PO_best = PO_MSE_avg, PO_MSE_pos, PO_MSE_vel, PO_MSE_acc, PO_KF, q

for q in q_values_AO:
    # System Model Initialization For different observation models
    model_AO = Linear_sysmdl.SystemModel(F=F_AO, Q=Q_AO * q * params.ratio_ca, H=H_AO, R=R_AO, T=T_decimated)
    # Initialize System Models with start conditions
    model_AO.InitSequence(m1x_0=m1x_0, m2x_0=m2x_0)

    [AO_MSE_avg, AO_MSE_pos, AO_MSE_vel, AO_MSE_acc, AO_KF] = KF_Pipeline.KF_Test1(model_AO, input_AO, target_per)

    if AO_MSE_avg_best is None or AO_MSE_avg_best > AO_MSE_avg:
        AO_MSE_avg_best, AO_MSE_pos_best, AO_MSE_vel_best, AO_MSE_acc_best, AO_KF_best, q_AO_best = AO_MSE_avg, AO_MSE_pos, AO_MSE_vel, AO_MSE_acc, AO_KF, q

for q in q_values_PAO:
    # System Model Initialization For different observation models
    model_PAO = Linear_sysmdl.SystemModel(F=F_PAO, Q=Q_PAO * q * params.ratio_ca, H=H_PAO, R=R_PAO, T=T_decimated)
    # Initialize System Models with start conditions
    model_PAO.InitSequence(m1x_0=m1x_0, m2x_0=m2x_0)

    [PAO_MSE_avg, PAO_MSE_pos, PAO_MSE_vel, PAO_MSE_acc, PAO_KF] = KF_Pipeline.KF_Test1(model_PAO, input_PAO, target_per)

    if PAO_MSE_avg_best is None or PAO_MSE_avg_best > PAO_MSE_avg:
        PAO_MSE_avg_best, PAO_MSE_pos_best, PAO_MSE_vel_best, PAO_MSE_acc_best, PAO_KF_best, q_PAO_best = PAO_MSE_avg, PAO_MSE_pos, PAO_MSE_vel, PAO_MSE_acc, PAO_KF, q

########################
### KF with q_search ###
########################

"""
q_values = torch.arange(0, 1000, 100)
for q in q_values:
    print("Testing q^2 values: ", q)
    Q = CV2D_mm.Q_dec*q

    # FULL OBSERVATION MODEL
    model_FO = Linear_sysmdl.SystemModel(F=F_decimated, Q=Q, H=H, R=R, T=T_decimated)
    model_FO.InitSequence(m1x_0=m1x_0, m2x_0=m2x_0)
    [FO_MSE_avg, FO_MSE_pos, FO_MSE_vel, FO_KF] = KF_Pipeline.KF_Test(model_FO, input_per, target_per)
    if FO_MSE_avg_best is None or FO_MSE_pos < FO_MSE_pos_best:
        FO_MSE_avg_best, FO_MSE_pos_best, FO_MSE_vel_best, FO_KF_best, q_FO_best = FO_MSE_avg, FO_MSE_pos, FO_MSE_vel, FO_KF, q

    # POS OBSERVATION MODEL
    model_PO = Linear_sysmdl.SystemModel(F=F_decimated, Q=Q, H=H_pos, R=R_pos, T=T_decimated)
    model_PO.InitSequence(m1x_0=m1x_0, m2x_0=m2x_0)
    [PO_MSE_avg, PO_MSE_pos, PO_MSE_vel, PO_KF] = KF_Pipeline.KF_Test(model_PO, input_per_pos, target_per)
    if PO_MSE_avg_best is None or PO_MSE_pos < PO_MSE_pos_best:
        PO_MSE_avg_best, PO_MSE_pos_best, PO_MSE_vel_best, PO_KF_best, q_PO_best = PO_MSE_avg, PO_MSE_pos, PO_MSE_vel, PO_KF, q

    # VEL OBSERVATION MODEL
    model_VO = Linear_sysmdl.SystemModel(F=F_decimated, Q=Q, H=H_vel, R=R_vel, T=T_decimated)
    model_VO.InitSequence(m1x_0=m1x_0, m2x_0=m2x_0)
    [VO_MSE_avg, VO_MSE_pos, VO_MSE_vel, VO_KF] = KF_Pipeline.KF_Test(model_VO, input_per_vel, target_per)
    if VO_MSE_avg_best is None or VO_MSE_pos < VO_MSE_pos_best:
        VO_MSE_avg_best, VO_MSE_pos_best, VO_MSE_vel_best, VO_KF_best, q_VO_best = VO_MSE_avg, VO_MSE_pos, VO_MSE_vel, VO_KF, q
"""

#################################
### Interpolated KALMANFILTER ###
#################################
"""
print("Interpolated KALMAN FILTER RUNNING")
sys_model_KF_I = Linear_sysmdl.SystemModel(F=F, Q=Q, H=H, R=R, T=T)
sys_model_KF_I.InitSequence(m1x_0=m1x_0, m2x_0=m2x_0)
[MSE_state_I_dB_avg, MSE_KF_I_dB_avg, KG_I_array, KF_I_out] = KF_Pipeline.KF_inter_Test(sys_model_KF_I, input_per, target)
"""



################
### PLOTTING ###
################

sample = 3
print("Plotting Sample ", sample)
ratio = params.ratio_cv

# TIME
time = torch.arange(0, T * params.T_cv, params.T_cv)
time_dec = torch.arange(0, T * params.T_cv, params.T_cv * ratio)

# POSITION PLOTS
x_GT = target[sample, 0, :]
y_GT = target[sample, 3, :]

x_per = target_per[sample, 0, :]
y_per = target_per[sample, 3, :]

x_per_input = input_per[sample, 0, :]
y_per_input = input_per[sample, 3, :]

x_KF_FO = FO_KF_best[sample, 0, :]
y_KF_FO = FO_KF_best[sample, 3, :]

x_KF_PO = PO_KF_best[sample, 0, :]
y_KF_PO = PO_KF_best[sample, 3, :]

x_KF_AO = AO_KF_best[sample, 0, :]
y_KF_AO = AO_KF_best[sample, 3, :]

x_KF_PAO = PAO_KF_best[sample, 0, :]
y_KF_PAO = PAO_KF_best[sample, 3, :]


# VELOCITY PLOTS
vx_GT = target[sample, 1, :]
vy_GT = target[sample, 4, :]

vx_per = target_per[sample, 1, :]
vy_per = target_per[sample, 4, :]

vx_per_input = input_per[sample, 1, :]
vy_per_input = input_per[sample, 4, :]

vx_KF_FO = FO_KF_best[sample, 1, :]
vy_KF_FO = FO_KF_best[sample, 4, :]

vx_KF_PO = PO_KF_best[sample, 1, :]
vy_KF_PO = PO_KF_best[sample, 4, :]

vx_KF_AO = AO_KF_best[sample, 1, :]
vy_KF_AO = AO_KF_best[sample, 4, :]

vx_KF_PAO = PAO_KF_best[sample, 1, :]
vy_KF_PAO = PAO_KF_best[sample, 4, :]


# ACCELERATION PLOTS
ax_GT = target[sample, 2, :]
ay_GT = target[sample, 5, :]

ax_per = target_per[sample, 2, :]
ay_per = target_per[sample, 5, :]

ax_per_input = input_per[sample, 2, :]
ay_per_input = input_per[sample, 5, :]

ax_KF_FO = FO_KF_best[sample, 2, :]
ay_KF_FO = FO_KF_best[sample, 5, :]

ax_KF_PO = PO_KF_best[sample, 2, :]
ay_KF_PO = PO_KF_best[sample, 5, :]

ax_KF_AO = AO_KF_best[sample, 2, :]
ay_KF_AO = AO_KF_best[sample, 5, :]

ax_KF_PAO = PAO_KF_best[sample, 2, :]
ay_KF_PAO = PAO_KF_best[sample, 5, :]



# Plotting parameters
lw_traj = 3.0
lw_vel = 1.0
lw_imp = 0.5
alpha = 0.8
dpi = 500




################
### FIGURE 1 ###
################
print("Figure 1")
file_name = plots_path + 'Figure_1.png'
fig, axs = plt.subplots(2, 2, figsize=(30, 15))

# GT
axs[0, 0].plot(x_GT, y_GT, 'g--', linewidth=lw_traj, alpha=alpha)
axs[0, 0].plot(x_per_input, y_per_input, color='red', linestyle=' ', marker='o', linewidth=lw_traj, alpha=alpha)
axs[0, 0].set_title("Trajectory Ground Truth")
axs[0, 0].legend(['Ground Truth', 'noisy observation'], fontsize = 'medium')
axs[0, 0].set_xlabel("x position (m)")
axs[0, 0].set_ylabel("y position (m)")

axs[0, 1].plot(x_KF_PO, y_KF_PO, color='dodgerblue', linewidth=lw_traj, alpha=alpha)
axs[0, 1].plot(x_GT, y_GT, 'g--', linewidth=lw_traj, alpha=alpha)
axs[0, 1].set_title("KF with pos observation")
axs[0, 1].set_xlabel("x position (m)")
axs[0, 1].set_ylabel("y position (m)")

axs[1, 0].plot(x_KF_AO, y_KF_AO, color='indigo', linewidth=lw_traj, alpha=alpha)
axs[1, 0].plot(x_GT, y_GT, 'g--', linewidth=lw_traj, alpha=alpha)
axs[1, 0].set_title("KF with acceleration observation")
axs[1, 0].set_xlabel("x position (m)")
axs[1, 0].set_ylabel("y position (m)")

axs[1, 1].plot(x_KF_PAO, y_KF_PAO, color='mediumvioletred', linewidth=lw_traj, alpha=alpha)
axs[1, 1].plot(x_GT, y_GT, 'g--', linewidth=lw_traj, alpha=alpha)
axs[1, 1].set_title("KF with position and acceleration observation")
axs[1, 1].set_xlabel("x position (m)")
axs[1, 1].set_ylabel("y position (m)")
fig.tight_layout()
plt.savefig(file_name, dpi=dpi)
plt.close('all')

################
### FIGURE 2 ###
################
print("Figure 2")
file_name = plots_path + 'Figure_2.png'
fig, axs = plt.subplots(1, 1, figsize=(30, 15))
axs.plot(x_GT, y_GT, 'g--', linewidth=lw_traj, alpha=alpha)
#axs.plot(x_per_input, y_per_input, color='red', linestyle=' ', marker='o', linewidth=lw_imp, alpha=alpha)
axs.plot(x_KF_PO, y_KF_PO, color='dodgerblue', linewidth=lw_traj, alpha=alpha)
axs.plot(x_KF_AO, y_KF_AO, color='indigo', linewidth=lw_traj, alpha=alpha)
axs.plot(x_KF_PAO, y_KF_PAO, color='mediumvioletred', linewidth=lw_traj, alpha=alpha)
axs.plot(x_KF_FO, y_KF_FO, color='gold', linewidth=lw_traj, alpha=alpha)
axs.legend(['Ground Truth', 'KF with position observation', 'KF with acceleration observation', 'KF with position and acceleration observation', 'KF with full observation' ], fontsize = 'large')
axs.set_title("Trajectories")
axs.set_xlabel("x position (m)")
axs.set_ylabel("y position (m)")
fig.tight_layout()
plt.savefig(file_name, dpi=dpi)
plt.close('all')

################
### FIGURE 3 ###
################
print("Figure 3")
# Everything we need for the plot
file_name = plots_path + 'Figure_3.png'
fig, axs = plt.subplots(2, 2, figsize=(30, 15))
# (p_x, p_y)

axs[0, 0].plot(x_GT, y_GT, 'g--', linewidth=lw_traj, alpha=alpha)
axs[0, 0].plot(x_per, y_per, color='black', linestyle=' ', marker='o', linewidth=lw_traj, alpha=alpha)
axs[0, 0].plot(x_per_input, y_per_input, color='red', linestyle=' ', marker='o', linewidth=lw_traj, alpha=alpha)
axs[0, 0].set_title("Trajectory Ground Truth")
axs[0, 0].legend(['Ground Truth', 'Decimated Ground Truth', 'noisy observations'], fontsize = 'medium')
axs[0, 0].set_xlabel("x position (m)")
axs[0, 0].set_ylabel("y position (m)")


axs[0, 1].plot(x_KF_PO, y_KF_PO, color='dodgerblue', linewidth=lw_traj, alpha=alpha)
axs[0, 1].plot(x_GT, y_GT, 'g--', linewidth=lw_traj, alpha=alpha)
axs[0, 1].set_title("KF with position observation")
axs[0, 1].legend(['KF with position observation', 'Ground Truth'], fontsize = 'medium')
axs[0, 1].set_xlabel("x position (m)")
axs[0, 1].set_ylabel("y position (m)")


axs[1, 0].plot(x_KF_AO, y_KF_AO, color='indigo', linewidth=lw_traj, alpha=alpha)
axs[1, 0].plot(x_GT, y_GT, 'g--', linewidth=lw_traj, alpha=alpha)
axs[1, 0].set_title("Trajectory KF with acceleration observation")
axs[1, 0].legend(['KF with acceleration observation', 'Ground Truth'], fontsize = 'medium')
axs[1, 0].set_xlabel("x position (m)")
axs[1, 0].set_ylabel("y position (m)")


axs[1, 1].plot(x_KF_PAO, y_KF_PAO, color='mediumvioletred', linewidth=lw_traj, alpha=alpha)
axs[1, 1].plot(x_GT, y_GT, 'g--', linewidth=lw_traj, alpha=alpha)
axs[1, 1].set_title("Trajectory KF with position and acceleration observation")
axs[1, 1].legend(['KF with position and acceleration observation', 'Ground Truth'], fontsize = 'medium')
axs[1, 1].set_xlabel("x position (m)")
axs[1, 1].set_ylabel("y position (m)")

fig.tight_layout()
plt.savefig(file_name, dpi=dpi)
plt.close('all')


################
### FIGURE 4 ###
################
print("Figure 4")
# Everything we need for the plot
file_name = plots_path + 'Figure_4.png'
fig, axs = plt.subplots(2, 2, figsize=(30, 15))

# (v_x, v_y)
axs[0, 0].plot(time, vx_GT, color='green', linestyle='--', linewidth=lw_vel, alpha=alpha)
axs[0, 0].plot(time_dec, vx_per_input, color='red', linestyle=' ', marker='o', linewidth=lw_vel, alpha=alpha)
axs[0, 0].set_title("Ground Truth Velocity")
axs[0, 0].legend(['Ground Truth', 'Noisy Observation'] , fontsize = 'medium')
axs[0, 0].set_xlabel("time (s)")
axs[0, 0].set_ylabel("velocity in x (m/s)")

axs[0, 1].plot(time, vx_GT, color='green', linestyle='--', linewidth=lw_vel, alpha=alpha)
axs[0, 1].plot(time_dec, vx_KF_PO, color='dodgerblue', linewidth=lw_vel, alpha=alpha)
axs[0, 1].set_title("KF with position observation")
axs[0, 1].legend(['Ground Truth', 'KF with position observation'], fontsize = 'medium')
axs[0, 1].set_xlabel("time (s)")
axs[0, 1].set_ylabel("velocity in x (m/s)")

axs[1, 0].plot(time, vx_GT, color='green', linestyle='--', linewidth=lw_vel, alpha=alpha)
axs[1, 0].plot(time_dec, vx_KF_AO, color='indigo', linewidth=lw_vel, alpha=alpha)
axs[1, 0].set_title("KF with acceleration observation")
axs[1, 0].legend(['Ground Truth', 'KF with acceleration observation'], fontsize = 'medium')
axs[1, 0].set_xlabel("time (s)")
axs[1, 0].set_ylabel("velocity in x (m/s)")

axs[1, 1].plot(time, vx_GT, color='green', linestyle='--', linewidth=lw_vel, alpha=alpha)
axs[1, 1].plot(time_dec, vx_KF_PAO, color='mediumvioletred', linewidth=lw_vel, alpha=alpha)
axs[1, 1].set_title("KF with position and acceleration observation")
axs[1, 1].legend(['Ground Truth', 'KF with position and acceleration observation'] , fontsize = 'medium')
axs[1, 1].set_xlabel("time (s)")
axs[1, 1].set_ylabel("velocity in x (m/s)")


fig.tight_layout()
plt.savefig(file_name, dpi=dpi)
plt.close('all')

################
### FIGURE 5 ###
################
print("Figure 5")
# Everything we need for the plot
file_name = plots_path + 'Figure_5.png'
fig, axs = plt.subplots(2, 2, figsize=(30, 15))

# (v_x, v_y)
axs[0, 0].plot(time, vy_GT, color='green', linestyle='--', linewidth=lw_vel, alpha=alpha)
axs[0, 0].plot(time_dec, vy_per_input, color='red', linestyle=' ', marker='o', linewidth=lw_vel, alpha=alpha)
axs[0, 0].set_title("Ground Truth Velocity")
axs[0, 0].legend(['Ground Truth', 'Noisy Observation'] , fontsize = 'medium')
axs[0, 0].set_xlabel("time (s)")
axs[0, 0].set_ylabel("velocity in y (m/s)")

axs[0, 1].plot(time, vy_GT, color='green', linestyle='--', linewidth=lw_vel, alpha=alpha)
axs[0, 1].plot(time_dec, vy_KF_PO, color='dodgerblue', linewidth=lw_vel, alpha=alpha)
axs[0, 1].set_title("KF with position observation")
axs[0, 1].legend(['Ground Truth', 'KF with position observation'], fontsize = 'medium')
axs[0, 1].set_xlabel("time (s)")
axs[0, 1].set_ylabel("velocity in y (m/s)")

axs[1, 0].plot(time, vy_GT, color='green', linestyle='--', linewidth=lw_vel, alpha=alpha)
axs[1, 0].plot(time_dec, vy_KF_AO, color='indigo', linewidth=lw_vel, alpha=alpha)
axs[1, 0].set_title("KF with acceleration observation")
axs[1, 0].legend(['Ground Truth', 'KF with acceleration observation'], fontsize = 'medium')
axs[1, 0].set_xlabel("time (s)")
axs[1, 0].set_ylabel("velocity in y (m/s)")

axs[1, 1].plot(time, vy_GT, color='green', linestyle='--', linewidth=lw_vel, alpha=alpha)
axs[1, 1].plot(time_dec, vy_KF_PAO, color='mediumvioletred', linewidth=lw_vel, alpha=alpha)
axs[1, 1].set_title("KF with position and acceleration observation")
axs[1, 1].legend(['Ground Truth', 'KF with position and acceleration observation'] , fontsize = 'medium')
axs[1, 1].set_xlabel("time (s)")
axs[1, 1].set_ylabel("velocity in y (m/s)")


fig.tight_layout()
plt.savefig(file_name, dpi=dpi)
plt.close('all')


################
### FIGURE 6 ###
################
print("Figure 6")
# Everything we need for the plot
file_name = plots_path + 'Figure_6.png'
fig, axs = plt.subplots(2, 2, figsize=(30, 15))

# (v_x, v_y)
axs[0, 0].plot(time, ax_GT, color='green', linestyle='--', linewidth=lw_vel, alpha=alpha)
axs[0, 0].plot(time_dec, ax_per_input, color='red', linestyle=' ', marker='o', linewidth=lw_vel, alpha=alpha)
axs[0, 0].set_title("Ground Truth Acceleration")
axs[0, 0].legend(['Ground Truth', 'Noisy Observation'] , fontsize = 'medium')
axs[0, 0].set_xlabel("time (s)")
axs[0, 0].set_ylabel("acceleration in x (m/s^2)")

axs[0, 1].plot(time, ax_GT, color='green', linestyle='--', linewidth=lw_vel, alpha=alpha)
axs[0, 1].plot(time_dec, ax_KF_PO, color='dodgerblue', linewidth=lw_vel, alpha=alpha)
axs[0, 1].set_title("KF with position observation")
axs[0, 1].legend(['Ground Truth', 'KF with position observation'], fontsize = 'medium')
axs[0, 1].set_xlabel("time (s)")
axs[0, 1].set_ylabel("acceleration in x (m/s^2)")

axs[1, 0].plot(time, ax_GT, color='green', linestyle='--', linewidth=lw_vel, alpha=alpha)
axs[1, 0].plot(time_dec, ax_KF_AO, color='indigo', linewidth=lw_vel, alpha=alpha)
axs[1, 0].set_title("KF with acceleration observation")
axs[1, 0].legend(['Ground Truth', 'KF with acceleration observation'], fontsize = 'medium')
axs[1, 0].set_xlabel("time (s)")
axs[1, 0].set_ylabel("acceleration in x (m/s^2)")

axs[1, 1].plot(time, ax_GT, color='green', linestyle='--', linewidth=lw_vel, alpha=alpha)
axs[1, 1].plot(time_dec, ax_KF_PAO, color='mediumvioletred', linewidth=lw_vel, alpha=alpha)
axs[1, 1].set_title("KF with position and acceleration observation")
axs[1, 1].legend(['Ground Truth', 'KF with position and acceleration observation'] , fontsize = 'medium')
axs[1, 1].set_xlabel("time (s)")
axs[1, 1].set_ylabel("acceleration in x (m/s^2)")


fig.tight_layout()
plt.savefig(file_name, dpi=dpi)
plt.close('all')


################
### FIGURE 7 ###
################
print("Figure 7")
# Everything we need for the plot
file_name = plots_path + 'Figure_7.png'
fig, axs = plt.subplots(2, 2, figsize=(30, 15))

# (v_x, v_y)
axs[0, 0].plot(time, ay_GT, color='green', linestyle='--', linewidth=lw_vel, alpha=alpha)
axs[0, 0].plot(time_dec, ay_per_input, color='red', linestyle=' ', marker='o', linewidth=lw_vel, alpha=alpha)
axs[0, 0].set_title("Ground Truth Acceleration")
axs[0, 0].legend(['Ground Truth', 'Noisy Observation'] , fontsize = 'medium')
axs[0, 0].set_xlabel("time (s)")
axs[0, 0].set_ylabel("acceleration in y (m/s^2)")

axs[0, 1].plot(time, ay_GT, color='green', linestyle='--', linewidth=lw_vel, alpha=alpha)
axs[0, 1].plot(time_dec, ay_KF_PO, color='dodgerblue', linewidth=lw_vel, alpha=alpha)
axs[0, 1].set_title("KF with position observation")
axs[0, 1].legend(['Ground Truth', 'KF with position observation'], fontsize = 'medium')
axs[0, 1].set_xlabel("time (s)")
axs[0, 1].set_ylabel("acceleration in y (m/s^2)")

axs[1, 0].plot(time, ay_GT, color='green', linestyle='--', linewidth=lw_vel, alpha=alpha)
axs[1, 0].plot(time_dec, ay_KF_AO, color='indigo', linewidth=lw_vel, alpha=alpha)
axs[1, 0].set_title("KF with acceleration observation")
axs[1, 0].legend(['Ground Truth', 'KF with acceleration observation'], fontsize = 'medium')
axs[1, 0].set_xlabel("time (s)")
axs[1, 0].set_ylabel("acceleration in y (m/s^2)")

axs[1, 1].plot(time, ay_GT, color='green', linestyle='--', linewidth=lw_vel, alpha=alpha)
axs[1, 1].plot(time_dec, ay_KF_PAO, color='mediumvioletred', linewidth=lw_vel, alpha=alpha)
axs[1, 1].set_title("KF with position and acceleration observation")
axs[1, 1].legend(['Ground Truth', 'KF with position and acceleration observation'] , fontsize = 'medium')
axs[1, 1].set_xlabel("time (s)")
axs[1, 1].set_ylabel("acceleration in y (m/s^2)")


fig.tight_layout()
plt.savefig(file_name, dpi=dpi)
plt.close('all')


######################
### SAVING RESULTS ###
######################
print("Saving results")
file = open(plots_path + "results_CA2D.txt", "w")
# Title
file.write("TEST RESULTS FILE:\n")

# Results
# FO
file.write("\n\n RESULTS FROM KF WITH FULL OBSERVATION:")
file.write("\npos MSE:" + str(FO_MSE_pos_best) + "[dB], " +
           "\nvel MSE:" + str(FO_MSE_vel_best) + "[dB], " +
            "\nacc MSE:" + str(FO_MSE_acc_best) + "[dB], " +
           "\ntotal MSE:" + str(FO_MSE_avg_best) + "[dB], ")

file.write("\n\n optimal variance of acceleration for KF_FO:" + str(q_FO_best))
file.write("\n\n optimal q^2 value for KF_FO:" + str(q_FO_best*params.ratio_cv*params.T_cv**2))

# PO
file.write("\n\n RESULTS FROM KF WITH POS OBSERVATION:")
file.write("\npos MSE:" + str(PO_MSE_pos_best) + "[dB], " +
           "\nvel MSE:" + str(PO_MSE_vel_best) + "[dB], " +
            "\nacc MSE:" + str(PO_MSE_acc_best) + "[dB], " +
           "\ntotal MSE:" + str(PO_MSE_avg_best) + "[dB], ")

file.write("\n\n optimal variance of acceleration for KF_PO:" + str(q_PO_best))
file.write("\n\n optimal q^2 value for KF_PO:" + str(q_PO_best*params.ratio_cv*params.T_cv**2))

# AO
file.write("\n\n RESULTS FROM KF WITH ACC OBSERVATION:")
file.write("\npos MSE:" + str(AO_MSE_pos_best) + "[dB], " +
           "\nvel MSE:" + str(AO_MSE_vel_best) + "[dB], " +
            "\nacc MSE:" + str(AO_MSE_acc_best) + "[dB], " +
           "\ntotal MSE:" + str(AO_MSE_avg_best) + "[dB], ")

file.write("\n\n optimal variance of acceleration for KF_VO:" + str(q_AO_best))
file.write("\n\n optimal q^2 value for KF_VO:" + str(q_AO_best*params.ratio_cv*params.T_cv**2))

# PAO
file.write("\n\n RESULTS FROM KF WITH POS AND ACC OBSERVATION:")
file.write("\npos MSE:" + str(PAO_MSE_pos_best) + "[dB], " +
           "\nvel MSE:" + str(PAO_MSE_vel_best) + "[dB], " +
            "\nacc MSE:" + str(PAO_MSE_acc_best) + "[dB], " +
           "\ntotal MSE:" + str(PAO_MSE_avg_best) + "[dB], ")

file.write("\n\n optimal variance of acceleration for KF_VO:" + str(q_PAO_best))
file.write("\n\n optimal q^2 value for KF_VO:" + str(q_PAO_best*params.ratio_cv*params.T_cv**2))



loss_fn = nn.MSELoss(reduction='mean')
MSE_total = loss_fn(input_per[:, :, :], target_per[:, :, :]).item()
MSE_pos = loss_fn(input_per[:, 0:3:2, :], target_per[:, 0:3:2, :]).item()
MSE_vel = loss_fn(input_per[:, 1:4:2, :], target_per[:, 1:4:2, :]).item()
MSE_acc = loss_fn(input_per[:, 2:6:3, :], target_per[:, 2:6:3, :]).item()

MSE_total_dB = 10*np.log10(MSE_total)
MSE_pos_dB = 10*np.log10(MSE_pos)
MSE_vel_dB = 10*np.log10(MSE_vel)
MSE_acc_dB = 10*np.log10(MSE_acc)


file.write("\n\n MSE of noisy observation and GT:\n")
file.write("\npos MSE:" + str(MSE_pos_dB) + "[dB], " +
            "\nvel MSE:" + str(MSE_vel_dB) + "[dB], " +
            "\nacc MSE:" + str(MSE_acc_dB) + "[dB], " +
           "\ntotal MSE:" + str(MSE_total_dB) + "[dB], ")

file.write("\n\n variance of acceleration for data generation: " + str(params.q_ca))
file.write("\n\n standard deviation of acceleration for data generation: " + str(params.q_ca**0.5))
file.write("\n\n q^2 value for data generation:" + str(params.q_ca*params.T_cv**2))
#file.write("\n\nKalman Gain :" + str(KGain))

# Date and time
# now = datetime.now()
# dd/mm/YY H:M:S
# dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
# file.write("\n\n\n\nSimulation Date & Time:"+ dt_string+"\n\n\n")

file.close()
print("results.txt successfully saved.")

pass

