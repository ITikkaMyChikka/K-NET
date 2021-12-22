import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from src.DECOUPLED_KF import data_loader
from src.DECOUPLED_KF import Encoder
from src.DECOUPLED_KF import Results_saver

from Linear_sysmdl import SystemModel
from Linear_KF import KalmanFilter




if torch.cuda.is_available():
    cuda0 = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
   cuda0 = torch.device("cpu")
   print("Running on the CPU")

#############
### PATHS ###
#############

results_path = "/Users/sidhu/Documents/ETH/Semester Project/K-NET/src/DECOUPLED_KF/Results/"
plots_path = "/Users/sidhu/Documents/ETH/Semester Project/K-NET/src/DECOUPLED_KF/Plots/"
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
 T_train, T_test, test_time, rtk_test] = data_loader.load_dataset(file)

train_input = train_input.to(cuda0,non_blocking = True)
cv_input = cv_input.to(cuda0,non_blocking = True)
test_input = test_input.to(cuda0,non_blocking = True)

train_target = train_target.to(cuda0,non_blocking = True)
cv_target = cv_target.to(cuda0,non_blocking = True)
test_target = test_target.to(cuda0,non_blocking = True)

train_init = train_init.to(cuda0,non_blocking = True)
cv_init = cv_init.to(cuda0,non_blocking = True)
test_init = test_init.to(cuda0,non_blocking = True)
print("data loaded successfully")

#####################
### KALMAN FILTER ###
#####################

####################
### Design Model ###
####################
# True model
dim = 2
t = 0.005 # 5ms
F = torch.tensor([[1.0, t], [0.0, 1.0]])
H = torch.eye(2)
r_x = 0.5730372729072545  # observation noise sqrt((error_vx + error_ay)/2), errors are not in dB (linear)
r_y = 0.90995209653925376
q = torch.linspace(start=0.15, end=0.6, steps=100) # We are doing here linesearch to find optimal q value
T = T_train # Actually this is unnecessary, since we only need T for training KalmanNet

# LOSS function (MSE)
loss_fn = nn.MSELoss(reduction='mean')

# Other parameters
N_T = test_input.size()[0] # Number of testing samples

# arrays to save best KF run with optimal q
MSE = None
MSE_dB_best = None
q_opt = None
x_opt = None
KGain_opt = None
MSE_y = None
MSE_dB_best_y = None
q_opt_y = None
y_opt = None
KGain_opt_y = None

# We do KF for different values of q and save the best result (line-search)
for q_value in q:
    print("Testing q_value:", q_value)

    # Define System Model: We basically give all the necessary Information to construct the State Space Model
    sys_model = SystemModel(F, q_value, H, r_x, T, T_test)
    sys_model_y = SystemModel(F, q_value, H, r_y, T, T_test)

    # Define KalmanFilter for each noise value q
    KF = KalmanFilter(sys_model)
    KF_y = KalmanFilter(sys_model_y)

    # Define arrays to save results
    # MSE_test_linear_arr = np.empty([N_T])
    MSE_test_linear_dim = torch.zeros((N_T, dim))
    x_out_array = torch.empty(N_T, dim, T_test)
    MSE_test_linear_dim_y = torch.zeros((N_T, dim))
    y_out_array = torch.empty(N_T, dim, T_test)

    for n_Test_sample in range(0, N_T):
        # Define arrays to save results
        KGain = None
        KGain_y = None
        vel_test = torch.zeros(2, T_test)
        vel_test_y = torch.zeros(2, T_test)

        # 1. Define x_0 (m1_0) and sigma_0 (m2_0)
        v_0 = torch.unsqueeze(test_init[n_Test_sample, :], dim=0).T  # ax, ay, yaw, vx, vy
        v_x_init = v_0[3][0]
        a_x_init = v_0[0][0]
        v_y_init = v_0[4][0]
        a_y_init = v_0[1][0]

        # Initial State vector
        m1_0 = torch.tensor([v_x_init, a_x_init]).to(cuda0)
        m1_0_y = torch.tensor([v_y_init,a_y_init]).to(cuda0)
        # Initial Covariance Matrix
        m2_0 = 0 * 0 * torch.eye(dim).to(cuda0)
        m2_0_y = 0 * 0 * torch.eye(dim).to(cuda0)

        # 2. Initialize system Model and then the Kalmanfilter with initial values
        sys_model.InitSequence(m1_0, m2_0)
        KF.InitSequence(sys_model.m1x_0, sys_model.m2x_0)
        sys_model_y.InitSequence(m1_0_y,m2_0_y)
        KF_y.InitSequence(sys_model_y.m1x_0, sys_model_y.m2x_0)

        # Initially we have to define the posterior (only so that the KF works)
        KF.m1x_posterior = KF.m1x_0
        KF.m2x_posterior = KF.m2x_0
        KF_y.m1x_posterior = KF_y.m1x_0
        KF_y.m2x_posterior = KF_y.m2x_0

        # 3.Prepare the first input to KF (measurements)
        y_mdl_tst = test_input[n_Test_sample, :, :]
        measurement = y_mdl_tst[:, 0:1]

        # 4.We iterate through the trajectory for each test sample (N_T) do Kalman Filtering
        for t in range(0, T_test):
            # From measurements downscale to vx and ax
            y = Encoder.get_sensor_reading(measurement, True)
            y_y = Encoder.get_sensor_reading(measurement, False)

            # Update does all the steps (prediction, K-GAIN, Innovation and correction)
            m1x_posterior, m2x_posterior = KF.Update(y)
            m1x_posterior_y, m2x_posterior_y = KF_y.Update(y_y)
            #MSE_KF_linear_arr[n_Test_sample] = loss_fn(KF.x, test_target[j, :, :]).item()

            vel_test[:, t] = m1x_posterior
            vel_test_y[:, t] = m1x_posterior_y

            # Obtain new measurements for next time step
            if (t + 1 != T_test):
                measurement = y_mdl_tst[:, t + 1:t + 2]
            else:
                # Save the Kalman gain for each test sample(N_T) at the end
                KGain = KF.KG
                KGain_y = KF_y.KG

        # We save the MSE for each test set (each trajectory)
        MSE_test_linear_dim[n_Test_sample, 0] = loss_fn(vel_test[0, :], test_target[n_Test_sample, 3, :]).item() # v_x
        MSE_test_linear_dim[n_Test_sample, 1] = loss_fn(vel_test[1, :], test_target[n_Test_sample, 0, :]).item() # a_x
        MSE_test_linear_dim_y[n_Test_sample, 0] = loss_fn(vel_test_y[0, :], test_target[n_Test_sample, 4, :]).item() # v_y
        MSE_test_linear_dim_y[n_Test_sample, 1] = loss_fn(vel_test_y[1, :], test_target[n_Test_sample, 1, :]).item() # a_y
        # We save the tracked states into x_out_array
        x_out_array[n_Test_sample, :, :] = vel_test
        y_out_array[n_Test_sample, :, :] = vel_test_y

    # We average our MSE over all testing samples (all trajectories) and calculate the dB
    MSE_test_linear_dim_avg = torch.mean(MSE_test_linear_dim, dim=0) # Actually does nothing since N_T = 1
    MSE_test_dB_dim_avg = 10 * torch.log10(MSE_test_linear_dim_avg)
    print("MSE Test x-axis : ", MSE_test_dB_dim_avg, " [dB]")

    MSE_test_linear_dim_avg_y = torch.mean(MSE_test_linear_dim_y, dim=0)
    MSE_test_dB_dim_avg_y = 10 * torch.log10(MSE_test_linear_dim_avg_y)
    print("MSE Test y-axis : ", MSE_test_dB_dim_avg_y, " [dB]")

    # Save the best run for optimal q_value
    if MSE_dB_best is None:
        MSE_dB_best = MSE_test_dB_dim_avg
        MSE = MSE_test_dB_dim_avg
        q_opt = q_value
        x_opt = x_out_array
        KGain_opt = KGain
    elif MSE_dB_best[0] > MSE_test_dB_dim_avg[0]:
        if (MSE_dB_best[0] + MSE_dB_best[1]) >= (MSE_test_dB_dim_avg[0] + MSE_test_dB_dim_avg[1]): # If the v_x error is smaller
           # We are mainly interested in the vel estimate. If the vx error is smaller we check if the total error is smaller than the best solution and then we update
            MSE_dB_best = MSE_test_dB_dim_avg
            MSE = MSE_test_dB_dim_avg
            q_opt = q_value
            x_opt = x_out_array
            KGain_opt = KGain
            """
        if MSE_dB_best[1] <= MSE_test_dB_dim_avg[1]:
            # Case where the MSE is better in vx & ax
            MSE_dB_best = MSE_test_linear_dim_avg
            MSE = MSE_test_dB_dim_avg
            q_opt = q_value
            x_opt = x_out_array
            KGain_opt = KGain
        """

    if MSE_dB_best_y is None:
        MSE_dB_best_y = MSE_test_dB_dim_avg_y
        MSE_y = MSE_test_dB_dim_avg_y
        q_opt_y = q_value
        y_opt = y_out_array
        KGain_opt_y = KGain_y
    elif MSE_dB_best_y[0] > MSE_test_dB_dim_avg_y[0]:
        if (MSE_dB_best_y[0] + MSE_dB_best_y[1]) >= (MSE_test_dB_dim_avg_y[0] + MSE_test_dB_dim_avg_y[1]): # If the v_y error is smaller
            MSE_dB_best_y = MSE_test_dB_dim_avg_y
            MSE_y = MSE_test_dB_dim_avg_y
            q_opt_y = q_value
            y_opt = y_out_array
            KGain_opt_y = KGain_y
            """
        if MSE_dB_best[1] <= MSE_test_dB_dim_avg_y[1]:
            MSE_dB_best_y = MSE_test_linear_dim_avg_y
            MSE_y = MSE_test_dB_dim_avg_y
            q_opt_y = q_value
            y_opt = y_out_array
            KGain_opt_y = KGain_y
        """


print("optimal q value for x-axis:", q_opt)
print("Corresponding KGain (x-axis):", KGain_opt)
print("optimal q value for y-axis:", q_opt_y)
print("Corresponding KGain (y-axis):", KGain_opt_y)

################
### PLOTTING ###
################
Results_saver.plot_stuff(plots_path=plots_path, x_opt=x_opt, y_opt=y_opt, test_target=test_target, test_time=test_time,
                         T_test=T_test, test_input=test_input, rtk_test=rtk_test)
Results_saver.generate_results_file(results_path= results_path, KGain=KGain_opt, q=q_opt, MSE=MSE,
                                    KGain_y=KGain_opt_y, q_opt_y = q_opt_y, MSE_y=MSE_y)
