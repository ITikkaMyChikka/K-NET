import numpy
import torch
import torch.nn as nn
import math



torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
from datetime import datetime

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import numpy as np

# Imports from Adria_KN
import pipeline
import models
import model_parameters as params
import torch.nn.functional as func

# For Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt

def custom_loss(output, target):
    # Input: [mxT]
    # Tries to normalize each output based on its range
    loss_MSE = nn.MSELoss(reduction='mean')
    LOSS = loss_MSE(output,target)
    return LOSS

def model_rpm(rpm, new_st):
    ## Steering angle to rads
    new_st = new_st * math.pi / 180

    ## Define inputs
    # Steering of previous timestep is needed for propagation
    # Updated at end of function and stored for next run
    global st
    try:
        st + 0
    except:
        st = new_st
        print("Old_st not working")

    # rpm is 4x1 (RPM of each wheel)
    # Convert to angular velocities
    rps = rpm * (2 * math.pi / 60)
    # Vehicle slow or fast?
    threshold_rpm = 30000

    if all(rpm < threshold_rpm):
        wheelspeed_update = True
    else:
        wheelspeed_update = False

    z_ws = torch.zeros(3, 1)

    if wheelspeed_update:
        # Covert to wheel velocities
        vel = (rps / params.gr_w) * (params.r_tyre * torch.ones(4, 1))[0, 0]
        ang_w = torch.tensor(
            [st / params.gr_s - params.toe[0, 0], st / params.gr_s + params.toe[0, 1], -params.toe[0, 2],
             params.toe[0, 3]])
        ang = torch.mean(ang_w[0:2]).item()
        vx = vel
        vx[0:2, :] = vx[0:2, :] * math.cos(ang)

        # vx
        z_ws[1, 0] = torch.mean(vx)

        yawrate = ((vx[1, 0] - vx[0, 0]) / params.tw + (vx[3, 0] - vx[2, 0]) / params.tw) / 2
        z_ws[0, 0] = yawrate
        # vy
        z_ws[2, 0] = yawrate * params.b

    return z_ws[:, :]


def get_sensor_reading(measurement):
    ## In this implementation we assume that measurement is the following:
    # [ax_imu(0),ay_imu(1),az_imu(2),dyaw_imu(3),ax_ins(4),ay_ins(5),az_ins(6),dyaw_ins(7),rpm_rl(8),rpm_rr(9),rpm_fl(10),rpm_fr(11),4xtm(12,13,14,15),sa(16)]

    # Step 1: obtain a_x through the IMU and INS
    y_imu = models.model_imu(measurement[0:4, :])
    y_ins = models.model_ins(measurement[4:8, :])
    # Average the IMU readings
    y_acc = (0.5 * y_imu + 0.5 * y_ins)
    acc = y_acc.reshape(3, )

    # Step 2: Obtain v_x through the RPM readings
    y_ws = model_rpm(measurement[8:12, :], measurement[16, :])[:, :]
    # y_ws = [yaw_rpm, v_x, v_y]

    # Merge all results into 1 vector
    # z_full: [ax_imu, ay_imu, dyaw_imu, ax_ins, ay_ins, dyaw_ins, vx_rpm, vy_rpm, dyaw_rpm]
    z_full = torch.cat((y_imu, y_ins, y_ws), dim=0)

    return z_full, acc



#############
### PATHS ###
#############

results_path = "/Users/sidhu/Documents/ETH/Semester Project/K-NET/src/Naive_VE/Sim1/Results/"
plots_path = "/Users/sidhu/Documents/ETH/Semester Project/K-NET/src/Naive_VE/Sim1/Plots/"
data_path = "/Datasets/"
dataset_file = "Dataset_50.pickle"
datasets_combined = ["Dataset_20.pickle","Dataset_50.pickle","Dataset_80.pickle","Dataset_200.pickle"]


####################
### LOADING DATA ###
####################
print("Loading data...")
file = "/Users/sidhu/Documents/ETH/Semester Project/Adria_KN/utils_data/Datasets/Dataset_50.pickle"
[train_input, cv_input, test_input,
 train_target, cv_target, test_target,
 train_init, cv_init, test_init,
 T_train, T_test, test_time, rtk_test] = pipeline.load_dataset(file)

"""
train_input = train_input.to(cuda0,non_blocking = True)
cv_input = cv_input.to(cuda0,non_blocking = True)
test_input = test_input.to(cuda0,non_blocking = True)

train_target = train_target.to(cuda0,non_blocking = True)
cv_target = cv_target.to(cuda0,non_blocking = True)
test_target = test_target.to(cuda0,non_blocking = True)

train_init = train_init.to(cuda0,non_blocking = True)
cv_init = cv_init.to(cuda0,non_blocking = True)
test_init = test_init.to(cuda0,non_blocking = True)"""
print("data loaded successfully")


#####################
### KALMAN FILTER ###
#####################

# x = [v_x, a_x]
# y = [v_x, a_x] v_x calculated from rtk and a_x from both IMU's
T = 0.005 # 5ms
n_Test_sample = 0
N_T = test_input.size()[0]
MSE_test_linear_dim = np.zeros((N_T,2))
x_out_array = np.zeros((N_T, 2, T_test))
z_full = torch.empty(N_T,9, T_test)

for n_Test_sample in range(0, N_T):

    # Initialize KF
    f_x = KalmanFilter(dim_x=2, dim_z=2)

    # Assign initial value for state vector
    v_0 = torch.unsqueeze(test_init[n_Test_sample, :], dim=0).T # ax ay, yaw, vx, vy
    v_0 = v_0.detach().numpy()
    v_x_init = v_0[3][0]
    a_x_init = v_0[0][0]
    f_x.x = np.array([[v_x_init],    # velocity
                      [a_x_init]])     # acceleration

    # Assign State Evolution Matrix
    f_x.F = np.array([[1.,T],
                     [0.,1.]])

    #Assign Observation Model
    f_x.H = np.array([[1.,0],
                      [0.,1.]])

    # Set uncertainty to high at the beginning
    f_x.P *= 1000.

    # Assign the process & observation noise
    R = np.array([[1.,0],
                 [0.,1.]])
    f_x.R = R #* noise TODO: include observation noise

    Q = np.array([[(1/3)*T*T*T, (1/2)*T*T],
                     [(1/2)*T*T,T]])
    f_x.Q = Q #* noise TODO: include process noise

    # Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)

    # We get the first observation data
    state_vector = np.zeros((2,T_test))
    y_mdl_tst = test_input[n_Test_sample, :, :]
    measurement = y_mdl_tst[:,0:1]

    # we iterate over the trajectory
    for t in range(0, T_test):


        #INit: y_mdl_tst[:, 0:1]
        # Running KF
        #z_full[n_Test_sample,:,t+1:t+2]
        z_ful, acc = get_sensor_reading(measurement) # z = [v_x, a_x] We can get ax from both IMU's how about vx
        v_x = z_ful.detach().numpy()[6,:] #z_full.detach().numpy()[n_Test_sample,6,t+1:t+2] #z_full.detach().numpy()[6,:]
        a_x_imu_1 = z_ful.detach().numpy()[0]# z_full.detach().numpy()[n_Test_sample,0,t+1:t+2]#
        a_x_imu_2 = z_ful.detach().numpy()[3] #z_full.detach().numpy()[n_Test_sample,3,t+1:t+2]#
        a_x = (a_x_imu_1+a_x_imu_2)/2
        z = np.array([v_x,    # position
                     a_x])  #np.array([v_x,    # velocity_rpm
                     #a_x])    # acceleration_imus
        f_x.predict()
        f_x.update(z)
        state_vector[0, t:t+1] = f_x.x[0] # v_x
        state_vector[1, t:t+1] = f_x.x[1] # a_x


        if (t + 1 != T_test):
            measurement = y_mdl_tst[:, t + 1:t + 2]

        # Output Kalmangain
        #print(f_x.K)

    x_out_array[n_Test_sample, :, :] = state_vector

    # Loss over separate dimensions
    loss_fn = custom_loss
    MSE_test_linear_dim[n_Test_sample, 0] = (np.square(state_vector[0,:] - test_target.detach().numpy()[n_Test_sample, 0, :])).mean(axis=None) # v_x
    MSE_test_linear_dim[n_Test_sample, 0] = (np.square(state_vector[1, :] - test_target.detach().numpy()[n_Test_sample, 3, :])).mean(axis=None) # a_x

    # Plotting
    # What I need:
    # state_KNet: x_out_array[0,:,plot_start:plot_limit]
    # state_MKF: test_target[0,:,plot_start:plot_limit]
    # time: test_time[0,0,plot_start:plot_limit]
    # sensors: y_processed[0,:,plot_start:plot_limit]
    file_name = plots_path + 'trajectories_0.png'
    fig, axs = plt.subplots(2, 3, figsize=(30, 15))
    lw_sens = 0.3
    lw_imp = 0.9
    fs = 'x-large'
    dpi = 500
    plot_limit = round(T_test)
    plot_start = round(0)

    state_KF = x_out_array[0,:,plot_start:plot_limit]
    state_MKF = test_target.detach()[0,:,plot_start:plot_limit]
    time = test_time.detach()[0,0,plot_start:plot_limit]
    sensors = z_full.detach()[0,:,plot_start:plot_limit]


    # a_x
    axs[0, 0].plot(time, state_KF[1, :], 'b', linewidth=lw_imp) # changed from state_KF[0] to state_KF[1]
    axs[0, 0].plot(time, state_MKF[0, :], 'r', linewidth=lw_imp)
    axs[0, 0].plot(time, sensors[0, :], 'g', linewidth=lw_sens)  # IMU
    axs[0, 0].plot(time, sensors[3, :], 'y', linewidth=lw_sens)  # INS
    axs[0, 0].set_title("a_x(t)")
    axs[0, 0].legend(('KNet', 'GT(MKF)', 'IMU', 'INS'), fontsize=fs)

    # v_x
    axs[1, 0].plot(time, state_KF[0, :], 'b', linewidth=lw_imp) # changed from state_KF[3, :] to state_KF[0]
    axs[1, 0].plot(time, state_MKF[3, :], 'r', linewidth=lw_imp)
    axs[1, 0].plot(time, sensors[7, :], 'm', linewidth=lw_sens)
    # axs[1, 0].plot(time, sensors[3,:], 'm', linewidth=lw_sens) # WS
    axs[1, 0].set_title("v_x(t)")
    axs[1, 0].legend(('KNet', 'GT(MKF)', 'WS'), fontsize=fs)

    fig.tight_layout()
    plt.savefig(file_name, dpi=dpi)
    print("Trajectory plot successfully saved")
    plt.close('all')

MSE_test_linear_dim_avg = np.mean(MSE_test_linear_dim)
MSE_test_dB_dim_avg = 10 * np.log10(MSE_test_linear_dim_avg)
print("MSE_test_dB_dim_avg was:")
print(MSE_test_dB_dim_avg)


