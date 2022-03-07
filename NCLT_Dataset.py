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
import src.Linear_models.KITTI_CA as CA
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


#########################
### GETTING NCLT DATA ###
#########################

date = '2013-01-10'
directory = '../NCLT_DATA/' + date + '/'
gpsConsumerGradeFile = 'gps.csv'
gpsRTKFile = 'gps_rtk.csv'
imuFile = 'ms25.csv'
imuEulerFile = 'ms25_euler.csv'
groundtruthFile = 'groundtruth.csv'
outputFile = '../data/data_' + date + '.mat'


#########################
### GROUND TRUTH NCLT ###
#########################

gt = np.loadtxt(directory + groundtruthFile, delimiter=",")
# NED (North, East Down)
gtTime = gt[:, 0] * 1e-6
x_GT = gt[:, 1]
y_GT = gt[:, 2]

###############################
### GPS Consumer Grade NCLT ###
###############################
gpsCG     = np.loadtxt(directory + gpsConsumerGradeFile, delimiter = ",")
gpsCGTime = gpsCG[:, 0] * 1e-6
gpsCGTime_s = gpsCG[:, 0]
latCG     = gpsCG[:, 3]
lngCG     = gpsCG[:, 4]

latCG0 = latCG[0]
lngCG0 = lngCG[0]

dLatCG = latCG - latCG0
dLngCG = lngCG - lngCG0

r = 6400000 # approx. radius of earth (m)
y_GPSCG = r * np.cos(latCG0) * np.sin(dLngCG)
x_GPSCG = r * np.sin(dLatCG)


####################
### GPS RTK NCLT ###
####################
gpsRTK     = np.loadtxt(directory + gpsRTKFile, delimiter = ",")
gpsRTKTime = gpsRTK[:, 0] * 1e-6
latRTK     = gpsRTK[:, 3]
lngRTK     = gpsRTK[:, 4]

latRTK0 = latRTK[0]
lngRTK0 = lngRTK[0]

dLatRTK = latRTK - latRTK0
dLngRTK = lngRTK - lngRTK0

r = 6400000 # approx. radius of earth (m)
y_GPSRTK = r * np.cos(latRTK0) * np.sin(dLngRTK)
x_GPSRTK = r * np.sin(dLatRTK)


################
### IMU NCLT ###
################
imu     = np.loadtxt(directory + imuFile, delimiter = ",")
imuTime = imu[:, 0] * 1e-6
# magX    = imu[:, 1]
# magY    = imu[:, 2]
# magZ    = imu[:, 3]
accelY  = imu[:, 4]  # TODO CHANGED THIS
accelX  = imu[:, 5]  # TODO CHANGED THIS
accelZ  = imu[:, 6]
gyroX   = imu[:, 7]
gyroY   = imu[:, 8]
gyroZ   = imu[:, 9]


"""
print("KITTI DATASET")
plots_path = "/Users/sidhu/Documents/ETH/Semester Project/K-NET/Results/"
file_name = plots_path + 'KITTI_DATASET.png'
lw_imp = 0.8
alpha = 0.8
dpi = 500

fig, axs = plt.subplots(1, 1, figsize=(30, 15))
axs.plot(x_GT, y_GT, 'g', linewidth=lw_imp, alpha=alpha)
axs.plot(x_GPSCG, y_GPSCG, 'b', linewidth=lw_imp, alpha=alpha)
axs.plot(x_GPSRTK, y_GPSRTK, 'c', linewidth=lw_imp, alpha=alpha)
axs.legend(['Ground Truth', 'GPS consumer grade', 'GPS RTK'])
axs.set_title("Trajectories")
axs.set_xlabel("x position (m)")
axs.set_ylabel("y position (m)")
fig.tight_layout()
plt.savefig(file_name, dpi=dpi)
plt.close('all')
"""


# NORMALIZING THE TIME STAMPS (SO WE START WITH 0.0)

GT_T_norm = gtTime - imuTime[0]
GPS_T_norm = gpsCGTime - imuTime[0]
IMU_T_norm = imuTime - imuTime[0]


# ROUND ALL VALUES TO 10^(-4) to be exact: Meaning the KF Filter will run with frequency delta_t = 0.0001
precision = 4
GT_T_rounded = [round(delta_t, precision) for delta_t in GT_T_norm]
GPS_T_rounded = [round(delta_t, precision) for delta_t in GPS_T_norm]
IMU_T_rounded = [round(delta_t, precision) for delta_t in IMU_T_norm]




# COMBINE TIME READINGS
# We check the timestamp of GPS and IMU
Sensor_time = []  # List containing all the times an observation arrives
Sensor_index = []  # List containing what kind of observation arrives
gps_index = 0
imu_index = 0
c1 = 0
c2 = 0
c3 = 0

while gps_index < len(GPS_T_rounded) or imu_index < len(IMU_T_rounded):

    if gps_index == len(GPS_T_rounded):

        Sensor_time.append(IMU_T_rounded[imu_index])  # Append time of IMU observation
        Sensor_index.append(1)  # 1: IMU observation
        imu_index += 1
        c2 += 1
    else:

        # If GPS value is next observation
        if GPS_T_rounded[gps_index] < IMU_T_rounded[imu_index]:
            Sensor_time.append(GPS_T_rounded[gps_index])  # Append time of GPS observation
            Sensor_index.append(0)  # 0: GPS observation
            gps_index += 1
            c1 += 1
        # If IMU is next observation
        elif IMU_T_rounded[imu_index] < GPS_T_rounded[gps_index]:
            Sensor_time.append(IMU_T_rounded[imu_index])  # Append time of IMU observation
            Sensor_index.append(1)  # 1: IMU observation
            imu_index += 1
            c2 += 1
        # IF IMU and GPS arrive at the same time
        elif IMU_T_rounded[imu_index] == GPS_T_rounded[gps_index]:
            Sensor_time.append(IMU_T_rounded[imu_index])
            Sensor_index.append(2)  # 2: IMU & GPS observation
            imu_index += 1
            gps_index += 1
            c3 += 1



# Convert the time stamp values to integers
sensor_time = []
for value in Sensor_time:
    sensor_time.append(int(value*10000))

GT_time = []
for value in GT_T_rounded:
    GT_time.append(int(value*10000))

GPS_time = []
for value in GPS_T_rounded:
    GPS_time.append(int(value*10000))


# We check if 2 timestamps have the same value, if yes we move the timestamp of one observation
for index, value in enumerate(sensor_time):
    if index == len(sensor_time)-1:
        index = index - 1
    if value == sensor_time[index+1]:
        sensor_time[index + 1] = value + 1












#########################
### CREATE INPUT DATA ###
#########################
# TRANSFORM THE OBSERVATION VALUES TO TENSOR

# INPUT_POS: pos measurement of GPSCG for x and y axis
# INPUT_ACC: acc measurement of IMU for x and y acis

# Ground Truth
x_GT = torch.from_numpy(x_GT)
y_GT = torch.from_numpy(y_GT)
target = torch.zeros((y_GT.shape[0], 2))  # (7186, 2), (#observations, (x, y))
target[:, 0] = x_GT
target[:, 1] = y_GT

# Consumer Grade GPS
print("Shapes of Observation: ", y_GPSCG.shape, x_GPSCG.shape, accelX.shape, accelY.shape)
print("Shapes of Time Stamps: ", )

x_GPSCG = torch.from_numpy(x_GPSCG)
y_GPSCG = torch.from_numpy(y_GPSCG)
input_pos = torch.zeros((y_GPSCG.shape[0], 2))  # (7186, 2), (#observations, (x, y))
input_pos[:, 0] = x_GPSCG
input_pos[:, 1] = y_GPSCG

# IMU Readings
x_acc = torch.from_numpy(accelX)
y_acc = torch.from_numpy(accelY)
input_acc = torch.zeros((accelX.shape[0], 2))  # (48324, 2), (#observations, (ax, ay))
input_acc[:, 0] = x_acc
input_acc[:, 1] = y_acc


x_bias = target[0, 0] - input_pos[0, 0]
target[:, 0] = target[:, 0] - x_bias
y_bias = target[0, 1] - input_pos[0, 1]
target[:, 1] = target[:, 1] - y_bias


# RTK GPS
x_GPSRTK = torch.from_numpy(x_GPSRTK)
y_GPSRTK = torch.from_numpy(y_GPSRTK)
x_acc = torch.from_numpy(accelX)
y_acc = torch.from_numpy(accelY)

input_pos_1 = torch.zeros((y_GPSCG.shape[0], 2))  # (7186, 2), (#observations, (x, y))
input_pos_1[:, 0] = x_GPSCG
input_pos_1[:, 1] = y_GPSCG

input_acc_1 = torch.zeros((accelX.shape[0], 2))  # (48324, 2), (#observations, (ax, ay))
input_acc_1[:, 0] = x_acc
input_acc_1[:, 1] = y_acc






###################################
### CALCULATE OBSERVATION NOISE ###
###################################

loss_fn = torch.nn.MSELoss(reduction='mean')
id = 5
loss = torch.zeros(1)
for index, t in enumerate(GT_time):
    if t < GPS_time[id] + 50 and t > GPS_time[id] - 50 :
        loss += loss_fn(target[index, :], input_pos[id, :])
        id += 1
loss = loss/id
loss_dB = 10 * torch.log10(loss)
print("#Compare GT to GPS ", id)
print("Number of GPS signals", len(GPS_time))
print("MSE loss linear of GPS and GT: ", loss)
print("MSE loss dB of GPS and GT: ", loss_dB)













####################
### SYSTEM MODEL ###
####################

# Matrices
F = CA.F
Q = CA.Q
H_PO = CA.H_PO
R_PO = CA.R_PO
H_AO = CA.H_AO
R_AO = CA.R_AO
H_PAO = CA.H_PAO
R_PAO = CA.R_PAO

# Calculate Trajectory Length
T_traj = sensor_time[-1] - sensor_time[0]  # total time of trajectory


#####################
### KALMAN FILTER ###
#####################

# System Model
#q_values = [ 10**(-7)]
q_values = [10**(-5), 10**(-7), 10**(-9)]
r_values = [10**(-2), 10**(0), 10**(-2)]
#r_values = [10**(0)]
KF_MSE_avg_best, KF_out_best = None, None
q_best = 0
r_best = 0
for r in r_values:
    for q in q_values:
        print("Running KF")
        sys_model= Linear_sysmdl.SystemModel_KITTI(F=F, Q=Q*q, H_PO=H_PO, R_PO=r*R_PO, H_AO=H_AO, R_AO=r*R_AO, H_PAO=H_PAO, R_PAO=r*R_PAO, T=T_traj)

        # Initialize
        m1x_0 = torch.tensor([target[0, 0], 0, 0, target[0, 1], 0, 0]).float()  # x_0 = [px, vx, ax, py, vy, ay]
        m2x_0 = 2000 * torch.eye(6, 6).float()
        sys_model.InitSequence(m1x_0=m1x_0, m2x_0=m2x_0)

        # Run Kalman Filter
        [KF_MSE_avg, KF_out] = KF_Pipeline.KF_Test_KITTI(sys_model, input_pos, input_acc, target, GT_time, sensor_time, Sensor_index)

        if KF_MSE_avg_best is None or KF_MSE_avg_best > KF_MSE_avg:
            KF_MSE_avg_best, KF_out_best = KF_MSE_avg, KF_out
            q_best = q
            r_best = r

print("BEST Q_VALUE = ", q_best)
print("BEST R_VALUE = ", r_best)
prev = KF_out_best[0, 0]
prev_y = KF_out_best[1, 1]


for i in range(1, KF_out_best.shape[1]):
    current = KF_out_best[0, i]
    current_y = KF_out_best[1, i]

    diff1 = torch.abs(current-prev)
    diff2 = torch.abs(current_y - prev_y)

    if diff1 > 10:
        KF_out_best[0, i] = prev
    if diff2 > 100:
        KF_out_best[1, i] = prev_y

    prev = KF_out_best[0, i]
    prev_y = KF_out_best[1, i]

























################
### PLOTTING ###
################


print("KITTI DATASET")
plots_path = "/Users/sidhu/Documents/ETH/Semester Project/K-NET/Results/"
file_name = plots_path + 'NCLT_DATASET.png'
lw_imp = 1.2
alpha = 0.8
dpi = 500

fig, axs = plt.subplots(1, 1, figsize=(30, 15))
axs.plot(target[:, 0], target[:, 1], 'g', linewidth=lw_imp, alpha=alpha)
axs.plot(input_pos[:, 0], input_pos[:, 1], 'b', linewidth=lw_imp, alpha=alpha)
axs.legend(['Ground Truth', 'GPS consumer grade'])
axs.set_title("Trajectories")
axs.set_xlabel("x position (m)")
axs.set_ylabel("y position (m)")
fig.tight_layout()
plt.savefig(file_name, dpi=dpi)
plt.close('all')

plots_path = "/Users/sidhu/Documents/ETH/Semester Project/K-NET/Results/"
file_name = plots_path + 'NCLT_DATASET_KF.png'
lw_imp = 1.2
alpha = 0.8
dpi = 500

fig, axs = plt.subplots(1, 1, figsize=(30, 15))
axs.plot(target[:, 0], target[:, 1], 'g', linewidth=lw_imp, alpha=alpha)
axs.plot(input_pos[:, 0], input_pos[:, 1], 'r', linewidth=0.4, alpha=alpha)
#axs.plot(x_GPSRTK, y_GPSRTK, 'c', linewidth=lw_imp, alpha=alpha)
axs.plot(KF_out_best[0, :], KF_out_best[3, :], 'b', linewidth=lw_imp, alpha=alpha)
axs.legend(['Ground Truth', 'GPS consumer grade', 'KF'])
axs.set_title("Trajectories")
axs.set_xlabel("x position (m)")
axs.set_ylabel("y position (m)")
fig.tight_layout()
plt.savefig(file_name, dpi=dpi)
plt.close('all')


plots_path = "/Users/sidhu/Documents/ETH/Semester Project/K-NET/Results/"
file_name = plots_path + 'NCLT_DATASET_KF_1.png'
lw_imp = 1.2
alpha = 0.8
dpi = 500

fig, axs = plt.subplots(1, 1, figsize=(30, 15))
axs.plot(target[:, 0], target[:, 1], 'g', linewidth=lw_imp, alpha=alpha)
axs.plot(input_pos[:, 0], input_pos[:, 1], 'r', linewidth=0.4, alpha=alpha)
#axs.plot(x_GPSRTK, y_GPSRTK, 'c', linewidth=lw_imp, alpha=alpha)
axs.plot(KF_out_best[0, :], KF_out_best[3, :], 'b', linewidth=lw_imp, alpha=alpha)
axs.legend(['Ground Truth', 'GPS consumer grade', 'KF'])
axs.set_title("Trajectories")
axs.set_xlabel("x position (m)")
axs.set_ylabel("y position (m)")
axs.set_xlim([-150, 100])
axs.set_ylim([-600, 150])
fig.tight_layout()
plt.savefig(file_name, dpi=dpi)
plt.close('all')


plots_path = "/Users/sidhu/Documents/ETH/Semester Project/K-NET/Results/"
file_name = plots_path + 'NCLT_DATASET_KF_2.png'
lw_imp = 1.2
alpha = 0.8
dpi = 500
time = torch.arange(0, T_traj/10000, 0.0001)

fig, axs = plt.subplots(1, 1, figsize=(30, 15))
axs.plot(time, KF_out_best[2], 'b', linewidth=lw_imp, alpha=alpha)
axs.plot(IMU_T_rounded, input_acc[:, 0], 'r', linewidth=0.4, alpha=alpha)
axs.legend(['KF', 'IMU'])
axs.set_title("Acceleration in X direction")
axs.set_xlabel("time in seconds")
axs.set_ylabel("acceleration in (m/s^2)")
fig.tight_layout()
plt.savefig(file_name, dpi=dpi)
plt.close('all')

plots_path = "/Users/sidhu/Documents/ETH/Semester Project/K-NET/Results/"
file_name = plots_path + 'NCLT_DATASET_KF_3.png'
lw_imp = 0.8
alpha = 0.8
dpi = 500
time = torch.arange(0, T_traj/10000, 0.0001)

fig, axs = plt.subplots(1, 1, figsize=(30, 15))
axs.plot(time, KF_out_best[5], 'b', linewidth=lw_imp, alpha=alpha)
axs.plot(IMU_T_rounded, input_acc[:, 0], 'r', linewidth=0.4, alpha=alpha)
axs.legend(['KF', 'IMU'])
axs.set_title("Acceleration in Y direction")
axs.set_xlabel("time in seconds")
axs.set_ylabel("acceleration in (m/s^2)")
fig.tight_layout()
plt.savefig(file_name, dpi=dpi)
plt.close('all')
