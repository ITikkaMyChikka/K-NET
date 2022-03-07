###############
### IMPORTS ###
###############
# Generic Imports
from datetime import datetime
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import numpy as np
import os
# KalmanNet_TSP imports
from src.KalmanNet_TSP import Linear_sysmdl
from src.KalmanNet_TSP import KF_Pipeline
from filterpy.kalman import KalmanFilter
# My imports
import src.Linear_models.CV2D_mm as CV
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

# data generation
F_gen = CV.F_gen
Q_gen = CV.Q_gen
H_gen = CV.H_gen
R_gen = CV.R_gen

# Full Observation
F_FO = CV.F_FO
Q_FO = CV.Q_FO
H_FO = CV.H_FO
R_FO = CV.R_FO

# Position Observation
F_PO = CV.F_PO
Q_PO = CV.Q_PO
H_PO = CV.H_PO
R_PO = CV.R_PO

# Velocity Observation
F_VO = CV.F_VO
Q_VO = CV.Q_VO
H_VO = CV.H_VO
R_VO = CV.R_VO



# Define Trajectory Length
T = 300000  # T-train defines the trajectory length for training&validation dataset
T_decimated = int(T / params.ratio_cv)

# Initialize SystemModel
sys_model = Linear_sysmdl.SystemModel(F=F_gen, Q=Q_gen, H=H_gen, R=R_gen, T=T)

# Initialize initial values for state vectors and covariance matrix
# Initial velocity corresponds to 18.0556m/s = 65km/h, then we start to either accelerate or decelerate
m1x_0 = torch.tensor([0, 2.0556, 0, 2.0556]).float()  # x_0 = [px, vx, py, vy]
m2x_0 = 0.0 * torch.eye(4, 4).float()  # P_0 = Identity * 2000, to show that we are very uncertain at the beginning
sys_model.InitSequence(m1x_0=m1x_0, m2x_0=m2x_0)

###############################
### GENERATE SYNTHETIC DATA ###
###############################
dataFileName = ['CA_synthetic']  # Naming the dataset
"""
# We only want to generate data if the folder is empty
#if os.path.isfile("./" + dataFileName[2]):
print("GENERATING DATA")
# Generating data
DataGen(sys_model, DatafolderName + dataFileName[5], T)

##############################
### LOADING SYNTHETIC DATA ###
##############################
print("Data Load")
[input, target] = torch.load(DatafolderName + dataFileName[5], map_location=cuda0)

print("testset size:", target.size())
print("testset input size:", input.size())

#######################
### DECIMATING DATA ###
#######################
print("DECIMATING DATA")
target_shape = target.shape
input_shape = input.shape
ratio = params.ratio_cv
target_dec = torch.zeros((target_shape[0], target_shape[1], int(target_shape[2]/ratio)))
input_dec = torch.zeros((input_shape[0], input_shape[1], int(input_shape[2] / ratio)))
for sample in range(0, input_shape[0]):
    for t in range(0, int(input_shape[2]/ratio)): # We make 2000 step (#EKF steps)
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
        mean = torch.zeros(R_vel.size()[0])
        er = np.random.multivariate_normal(mean, R_vel, 1)
        input_per[sample, :, t] = input_dec[sample, :, t] + er
        target_per[sample, :, t] = target_dec[sample, :, t]

        #noise_target = torch.randn(target_shape[1])
        #noise_input = torch.randn(input_shape[1])
        #target_per[sample, :, t] = target_dec[sample, :, t] + noise_target * perturbation
        #input_per[sample, :, t] = input_dec[sample, :, t] + noise_input * perturbation


torch.save([input, target, target_dec, input_dec, target_per, input_per], DatafolderName + dataFileName[5])
"""

####################
### LOADING DATA ###
####################
print("LOADING FINAL DATA")
print(dataFileName[0])
[input, target, target_dec, input_dec, target_per, input_per] = torch.load(DatafolderName + dataFileName[0], map_location=cuda0)


# data from CA model (dim=6), reduce to CV model
input = torch.cat((input[:, 0:2, :], input[:, 3:5, :]), 1)
target = torch.cat((target[:, 0:2, :], target[:, 3:5, :]), 1)
input_dec = torch.cat((input_dec[:, 0:2, :], input_dec[:, 3:5, :]), 1)
target_dec = torch.cat((target_dec[:, 0:2, :], target_dec[:, 3:5, :]), 1)
input_per = torch.cat((input_per[:, 0:2, :], input_per[:, 3:5, :]), 1)
target_per = torch.cat((target_per[:, 0:2, :], target_per[:, 3:5, :]), 1)

#####################
### POS ONLY DATA ###
#####################

input_PO = input_per[:, 0:3:2, :]


####################
### VEL ONLY DAT ###
####################

input_VO = input_per[:, 1:4:2, :]


####################
### KALMANFILTER ###
####################
print("KALMAN FILTER RUNNING")

# Initialize empty files to save results of KF
FO_MSE_avg_best, FO_MSE_pos_best, FO_MSE_vel_best, FO_KF_best, q_FO_best = None, None, None, None, None
PO_MSE_avg_best, PO_MSE_pos_best, PO_MSE_vel_best, PO_KF_best, q_PO_best = None, None, None, None, None
VO_MSE_avg_best, VO_MSE_pos_best, VO_MSE_vel_best, VO_KF_best, q_VO_best = None, None, None, None, None


########################
### KF with q_search ###
########################
q_values_FO = torch.arange(0, 0.01, 0.0001)
q_values_PO = torch.arange(0, 0.01, 0.0001)
q_values_VO = torch.arange(0, 0.001, 0.00001)

for q in q_values_FO:
    print("Testing q_value: ", q)
    # Define System Models
    model_FO = Linear_sysmdl.SystemModel(F=F_FO, Q=Q_FO * q * params.ratio_cv, H=H_FO, R=R_FO, T=T_decimated)
    #model_PO = Linear_sysmdl.SystemModel(F=F_PO, Q=Q_PO * q * params.ratio_cv, H=H_PO, R=R_PO, T=T_decimated)
    #model_VO = Linear_sysmdl.SystemModel(F=F_VO, Q=Q_VO * q * params.ratio_cv, H=H_VO, R=R_VO, T=T_decimated)

    # Initialize System Models with start conditions
    model_FO.InitSequence(m1x_0=m1x_0, m2x_0=m2x_0)
    #model_PO.InitSequence(m1x_0=m1x_0, m2x_0=m2x_0)
    #model_VO.InitSequence(m1x_0=m1x_0, m2x_0=m2x_0)

    # Running Kalman Filters
    [FO_MSE_avg, FO_MSE_pos, FO_MSE_vel, FO_KF] = KF_Pipeline.KF_Test(model_FO, input_per, target_per)
    #[PO_MSE_avg, PO_MSE_pos, PO_MSE_vel, PO_KF] = KF_Pipeline.KF_Test(model_PO, input_PO, target_per)
    #[VO_MSE_avg, VO_MSE_pos, VO_MSE_vel, VO_KF] = KF_Pipeline.KF_Test(model_VO, input_VO, target_per)

    # Compare and save best results
    if FO_MSE_avg_best is None or FO_MSE_pos_best > FO_MSE_pos:
        FO_MSE_avg_best, FO_MSE_pos_best, FO_MSE_vel_best, FO_KF_best, q_FO_best = FO_MSE_avg, FO_MSE_pos, FO_MSE_vel, FO_KF, q
    #if PO_MSE_avg_best is None or PO_MSE_pos_best > PO_MSE_pos:
    #    PO_MSE_avg_best, PO_MSE_pos_best, PO_MSE_acc_best, PO_KF_best, q_PO_best = PO_MSE_avg, PO_MSE_pos, PO_MSE_vel, PO_KF, q
    #if VO_MSE_avg_best is None or VO_MSE_pos_best > VO_MSE_pos:
    #    VO_MSE_avg_best, VO_MSE_pos_best, VO_MSE_acc_best, VO_KF_best, q_VO_best = VO_MSE_avg, VO_MSE_pos, VO_MSE_vel, VO_KF, q

for q in q_values_PO:
    print("Testing q_value: ", q)
    # Define System Models
    model_PO = Linear_sysmdl.SystemModel(F=F_PO, Q=Q_PO * q * params.ratio_cv, H=H_PO, R=R_PO, T=T_decimated)

    # Initialize System Models with start conditions
    model_PO.InitSequence(m1x_0=m1x_0, m2x_0=m2x_0)

    # Running Kalman Filters
    [PO_MSE_avg, PO_MSE_pos, PO_MSE_vel, PO_KF] = KF_Pipeline.KF_Test(model_PO, input_PO, target_per)

    # Compare and save best results
    if PO_MSE_avg_best is None or PO_MSE_pos_best > PO_MSE_pos:
        PO_MSE_avg_best, PO_MSE_pos_best, PO_MSE_vel_best, PO_KF_best, q_PO_best = PO_MSE_avg, PO_MSE_pos, PO_MSE_vel, PO_KF, q

for q in q_values_VO:
    print("Testing q_value: ", q)
    # Define System Models
    model_VO = Linear_sysmdl.SystemModel(F=F_VO, Q=Q_VO * q * params.ratio_cv, H=H_VO, R=R_VO, T=T_decimated)

    # Initialize System Models with start conditions
    model_VO.InitSequence(m1x_0=m1x_0, m2x_0=m2x_0)

    # Running Kalman Filters
    [VO_MSE_avg, VO_MSE_pos, VO_MSE_vel, VO_KF] = KF_Pipeline.KF_Test(model_VO, input_VO, target_per)

    # Compare and save best results
    if VO_MSE_avg_best is None or VO_MSE_pos_best > VO_MSE_pos:
        VO_MSE_avg_best, VO_MSE_pos_best, VO_MSE_vel_best, VO_KF_best, q_VO_best = VO_MSE_avg, VO_MSE_pos, VO_MSE_vel, VO_KF, q

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
ratio = params.ratio_cv

# TIME
time = torch.arange(0, T * params.T_cv, params.T_cv)
time_dec = torch.arange(0, T * params.T_cv, params.T_cv * ratio)

# POSITION PLOTS
x_GT = target[sample, 0, :]  # start_plot:end_plot]
y_GT = target[sample, 2, :]

x_per = target_per[sample, 0, :]  # start_plot_dec:end_plot_dec]
y_per = target_per[sample, 2, :]

x_per_input = input_per[sample, 0, :]
y_per_input = input_per[sample, 2, :]

x_KF_FO = FO_KF_best[sample, 0, :]
y_KF_FO = FO_KF_best[sample, 2, :]

x_KF_PO = PO_KF_best[sample, 0, :]
y_KF_PO = PO_KF_best[sample, 2, :]

x_KF_VO = VO_KF_best[sample, 0, :]
y_KF_VO = VO_KF_best[sample, 2, :]


# VELOCITY PLOTS
vx_GT = target[sample, 1, :]  # start_plot:end_plot]
vy_GT = target[sample, 3, :]

vx_per = target_per[sample, 1, :]  # start_plot_dec:end_plot_dec]
vy_per = target_per[sample, 3, :]

vx_per_input = input_per[sample, 1, :]  # start_plot:end_plot]
vy_per_input = input_per[sample, 3, :]

vx_KF_FO = FO_KF_best[sample, 1, :]
vy_KF_FO = FO_KF_best[sample, 3, :]

vx_KF_PO = PO_KF_best[sample, 1, :]
vy_KF_PO = PO_KF_best[sample, 3, :]

vx_KF_VO = VO_KF_best[sample, 1, :]
vy_KF_VO = VO_KF_best[sample, 3, :]

# Plotting parameters
lw_traj = 3.0
lw_vel = 1.6
lw_obs = 1.0
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
axs[0, 0].plot(x_per_input, y_per_input, color='red', linestyle=' ', marker='o', linewidth=lw_obs, alpha=alpha)
axs[0, 0].set_title("Trajectory Ground Truth")
axs[0, 0].legend(['Ground Truth', 'noisy observation'] , fontsize = 'medium')
axs[0, 0].set_xlabel("x position (m)")
axs[0, 0].set_ylabel("y position (m)")

axs[0, 1].plot(x_KF_FO, y_KF_FO, color='dodgerblue', linewidth=lw_traj, alpha=alpha)
axs[0, 1].plot(x_GT, y_GT, 'g--', linewidth=lw_traj, alpha=alpha)
axs[0, 1].set_title("KF with position and velocity observation")
axs[0, 1].set_xlabel("x position (m)")
axs[0, 1].set_ylabel("y position (m)")

axs[1, 0].plot(x_KF_PO, y_KF_PO, color='indigo', linewidth=lw_traj, alpha=alpha)
axs[1, 0].plot(x_GT, y_GT, 'g--', linewidth=lw_traj, alpha=alpha)
axs[1, 0].set_title("KF with position observation")
axs[1, 0].set_xlabel("x position (m)")
axs[1, 0].set_ylabel("y position (m)")

axs[1, 1].plot(x_KF_VO, y_KF_VO, color='mediumvioletred', linewidth=lw_traj, alpha=alpha)
axs[1, 1].plot(x_GT, y_GT, 'g--', linewidth=lw_traj, alpha=alpha)
axs[1, 1].set_title("KF with velocity observation")
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
axs.plot(x_KF_FO, y_KF_FO, color='dodgerblue', linewidth=lw_traj, alpha=alpha)
axs.plot(x_KF_PO, y_KF_PO, color='indigo', linewidth=lw_traj, alpha=alpha)
axs.plot(x_KF_VO, y_KF_VO, color='mediumvioletred', linewidth=lw_traj, alpha=alpha)
axs.legend(['Ground Truth', 'KF with position and velocity observation', 'KF with position observation', 'KF with velocity observation' ], fontsize = 'large')
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
axs[0, 0].plot(x_per_input, y_per_input, color='red', linestyle=' ', marker='o', linewidth=lw_obs, alpha=alpha)
axs[0, 0].set_title("Trajectory Ground Truth")
axs[0, 0].legend(['Ground Truth', 'Decimated Ground Truth', 'noisy observations'], fontsize = 'medium')
axs[0, 0].set_xlabel("x position (m)")
axs[0, 0].set_ylabel("y position (m)")


axs[0, 1].plot(x_KF_FO, y_KF_FO, color='dodgerblue', linewidth=lw_traj, alpha=alpha)
axs[0, 1].plot(x_GT, y_GT, 'g--', linewidth=lw_traj, alpha=alpha)
axs[0, 1].set_title("KF with full observation")
axs[0, 1].legend(['KF with full observation', 'Ground Truth'], fontsize = 'medium')
axs[0, 1].set_xlabel("x position (m)")
axs[0, 1].set_ylabel("y position (m)")


axs[1, 0].plot(x_KF_PO, y_KF_PO, color='indigo', linewidth=lw_traj, alpha=alpha)
axs[1, 0].plot(x_GT, y_GT, 'g--', linewidth=lw_traj, alpha=alpha)
axs[1, 0].set_title("Trajectory KF with position observation")
axs[1, 0].legend(['KF with position observation', 'Ground Truth'], fontsize = 'medium')
axs[1, 0].set_xlabel("x position (m)")
axs[1, 0].set_ylabel("y position (m)")


axs[1, 1].plot(x_KF_VO, y_KF_VO, color='mediumvioletred', linewidth=lw_traj, alpha=alpha)
axs[1, 1].plot(x_GT, y_GT, 'g--', linewidth=lw_traj, alpha=alpha)
axs[1, 1].set_title("Trajectory KF with velocity observation")
axs[1, 1].legend(['KF with velocity observation', 'Ground Truth'] , fontsize = 'medium')
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
axs[0, 0].plot(time_dec, vx_per_input, color='red', linestyle=' ', marker='o', linewidth=lw_obs, alpha=alpha)
axs[0, 0].set_title("Ground Truth Velocity")
axs[0, 0].legend(['Ground Truth', 'Noisy Observation'] , fontsize = 'medium')
axs[0, 0].set_xlabel("time (s)")
axs[0, 0].set_ylabel("velocity in x (m/s)")

axs[0, 1].plot(time, vx_GT, color='green', linestyle='--', linewidth=lw_vel, alpha=alpha)
axs[0, 1].plot(time_dec, vx_KF_FO, color='dodgerblue', linewidth=lw_vel, alpha=alpha)
axs[0, 1].set_title("KF with full observation")
axs[0, 1].legend(['Ground Truth', 'KF with full observation'], fontsize = 'medium')
axs[0, 1].set_xlabel("time (s)")
axs[0, 1].set_ylabel("velocity in x (m/s)")

axs[1, 0].plot(time, vx_GT, color='green', linestyle='--', linewidth=lw_vel, alpha=alpha)
axs[1, 0].plot(time_dec, vx_KF_PO, color='indigo', linewidth=lw_vel, alpha=alpha)
axs[1, 0].set_title("KF with position observation")
axs[1, 0].legend(['Ground Truth', 'KF with position observation'], fontsize = 'medium')
axs[1, 0].set_xlabel("time (s)")
axs[1, 0].set_ylabel("velocity in x (m/s)")

axs[1, 1].plot(time, vx_GT, color='green', linestyle='--', linewidth=lw_vel, alpha=alpha)
axs[1, 1].plot(time_dec, vx_KF_VO, color='mediumvioletred', linewidth=lw_vel, alpha=alpha)
axs[1, 1].set_title("KF with velocity observation")
axs[1, 1].legend(['Ground Truth', 'KF with velocity observation'] , fontsize = 'medium')
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
axs[0, 0].plot(time_dec, vy_per_input, color='red', linestyle=' ', marker='o', linewidth=lw_obs, alpha=alpha)
axs[0, 0].set_title("Ground Truth Velocity")
axs[0, 0].legend(['Ground Truth', 'Noisy Observation'], fontsize = 'medium')
axs[0, 0].set_xlabel("time (s)")
axs[0, 0].set_ylabel("velocity in y (m/s)")

axs[0, 1].plot(time, vy_GT, color='green', linestyle='--', linewidth=lw_vel, alpha=alpha)
axs[0, 1].plot(time_dec, vy_KF_FO, color='dodgerblue', linewidth=lw_vel, alpha=alpha)
axs[0, 1].set_title("KF with full observation")
axs[0, 1].legend(['Ground Truth', 'KF with full observation'], fontsize = 'medium')
axs[0, 1].set_xlabel("time (s)")
axs[0, 1].set_ylabel("velocity in y (m/s)")

axs[1, 0].plot(time, vy_GT, color='green', linestyle='--', linewidth=lw_vel, alpha=alpha)
axs[1, 0].plot(time_dec, vy_KF_PO, color='indigo', linewidth=lw_vel, alpha=alpha)
axs[1, 0].set_title("KF with position observation")
axs[1, 0].legend(['Ground Truth', 'KF with position observation'], fontsize = 'medium')
axs[1, 0].set_xlabel("time (s)")
axs[1, 0].set_ylabel("velocity in y (m/s)")

axs[1, 1].plot(time, vy_GT, color='green', linestyle='--', linewidth=lw_vel, alpha=alpha)
axs[1, 1].plot(time_dec, vy_KF_VO, color='mediumvioletred', linewidth=lw_vel, alpha=alpha)
axs[1, 1].set_title("KF with velocity observation")
axs[1, 1].legend(['Ground Truth', 'KF with velocity observation'] , fontsize = 'medium')
axs[1, 1].set_xlabel("time (s)")
axs[1, 1].set_ylabel("velocity in y (m/s)")


fig.tight_layout()
plt.savefig(file_name, dpi=dpi)
plt.close('all')


######################
### SAVING RESULTS ###
######################
print("Saving results")
file = open(plots_path + "results_CV2D.txt", "w")
# Title
file.write("TEST RESULTS FILE:\n")

# Results
# FO
file.write("\n\n RESULTS FROM KF WITH POS & VEL OBSERVATION:")
file.write("\npos MSE:" + str(FO_MSE_pos_best) + "[dB], " +
           "\nvel MSE:" + str(FO_MSE_vel_best) + "[dB], " +
           "\ntotal MSE:" + str(FO_MSE_avg_best) + "[dB], ")

file.write("\n\n optimal variance of acceleration for KF_FO:" + str(q_FO_best))
file.write("\n\n optimal q^2 value for KF_FO:" + str(q_FO_best*params.ratio_cv*params.T_cv**2))

# PO
file.write("\n\n RESULTS FROM KF WITH POS OBSERVATION:")
file.write("\npos MSE:" + str(PO_MSE_pos_best) + "[dB], " +
           "\nvel MSE:" + str(PO_MSE_vel_best) + "[dB], " +
           "\ntotal MSE:" + str(PO_MSE_avg_best) + "[dB], ")

file.write("\n\n optimal variance of acceleration for KF_PO:" + str(q_PO_best))
file.write("\n\n optimal q^2 value for KF_PO:" + str(q_PO_best*params.ratio_cv*params.T_cv**2))

# VO
file.write("\n\n RESULTS FROM KF WITH VEL OBSERVATION:")
file.write("\npos MSE:" + str(VO_MSE_pos_best) + "[dB], " +
           "\nvel MSE:" + str(VO_MSE_vel_best) + "[dB], " +
           "\ntotal MSE:" + str(VO_MSE_avg_best) + "[dB], ")

file.write("\n\n optimal variance of acceleration for KF_VO:" + str(q_VO_best))
file.write("\n\n optimal q^2 value for KF_VO:" + str(q_VO_best*params.ratio_cv*params.T_cv**2))



loss_fn = nn.MSELoss(reduction='mean')
MSE_total = loss_fn(input_per[:, :, :], target_per[:, :, :]).item()
MSE_pos = loss_fn(input_per[:, 0:3:2, :], target_per[:, 0:3:2, :]).item()
MSE_vel = loss_fn(input_per[:, 1:4:2, :], target_per[:, 1:4:2, :]).item()
MSE_total_dB = 10*np.log10(MSE_total)
MSE_pos_dB = 10*np.log10(MSE_pos)
MSE_vel_dB = 10*np.log10(MSE_vel)


file.write("\n\n MSE of noisy observation and GT:\n")
file.write("\npos MSE:" + str(MSE_pos_dB) + "[dB], " +
            "\nvel MSE:" + str(MSE_vel_dB) + "[dB], " +
           "\ntotal MSE:" + str(MSE_total_dB) + "[dB], ")

file.write("\n\n variance of acceleration for data generation: " + str(params.q_cv))
file.write("\n\n standard deviation of acceleration for data generation: " + str(params.q_cv**0.5))
file.write("\n\n q^2 value for data generation:" + str(params.q_cv*params.T_cv**2))
#file.write("\n\nKalman Gain :" + str(KGain))

# Date and time
# now = datetime.now()
# dd/mm/YY H:M:S
# dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
# file.write("\n\n\n\nSimulation Date & Time:"+ dt_string+"\n\n\n")

file.close()
print("results.txt successfully saved.")

pass

