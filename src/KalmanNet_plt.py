#import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import matplotlib as plt

from NN_parameters import N_Epochs

N_Epochs_plt = N_Epochs

# Legend
Klegend = ["KNet - Train", "KNet - CV", "KNet - Test", "Extended Kalman Filter", "Baseline"]
Klegend_partial = ["KNet - Train", "KNet - CV", "KNet - Test", "Extended Kalman Filter Full", "Baseline", "Extended Kalman Filter Partial"]
# Color
KColor = ['ro', 'yo', 'g-', 'b-', 'y-']
KColor_partial = ['ro', 'yo', 'g-', 'b-', 'y-', 'r-']

def NNPlot_train(MSE_test_linear_arr, MSE_test_dB_avg,
                 MSE_cv_dB_epoch, MSE_train_dB_epoch, 
                 file_path):

    N_Epochs_plt = np.shape(MSE_cv_dB_epoch)[0]

    ###########################
    ### Plot per epoch [dB] ###
    ###########################
    plt.pyplot.figure(figsize = (50, 20))

    x_plt = range(0, N_Epochs_plt)

    # Train
    y_plt1 = MSE_train_dB_epoch[range(0, N_Epochs_plt)]
    plt.pyplot.plot(x_plt, y_plt1, KColor[0], label=Klegend[0])

    # CV
    y_plt2 = MSE_cv_dB_epoch[range(0, N_Epochs_plt)]
    plt.pyplot.plot(x_plt, y_plt2, KColor[1], label=Klegend[1])

    # KNet - Test
    y_plt3 = MSE_test_dB_avg * np.ones(N_Epochs_plt)
    plt.pyplot.plot(x_plt, y_plt3, KColor[2], label=Klegend[2])

    plt.pyplot.legend()
    plt.pyplot.xlabel('Number of Training Epochs', fontsize=16)
    plt.pyplot.ylabel('MSE Loss Value [dB]', fontsize=16)
    plt.pyplot.title('MSE Loss [dB] - per Epoch', fontsize=16)
    plt.pyplot.savefig(file_path + 'plt_model_test_dB')

    ####################
    ### dB Histogram ###
    ####################

    plt.pyplot.figure(figsize=(50, 20))
    sns.distplot(10 * np.log10(MSE_test_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color='g', label = 'KalmanNet')
    plt.pyplot.legend()
    plt.pyplot.title("Histogram [dB]")
    plt.pyplot.savefig(file_path + 'plt_hist_dB')

    print('End')

def KNPlot_test(MSE_KF_design_linear_arr, MSE_KF_data_linear_arr, MSE_KN_linear_arr):

    ####################
    ### dB Histogram ###
    ####################
    plt.pyplot.figure(figsize=(50, 20))
    sns.distplot(10 * np.log10(MSE_KN_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color='g', label = 'KalmanNet')
    sns.distplot(10 * np.log10(MSE_KF_design_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color= 'b', label = 'Extended Kalman Filter - design')
    sns.distplot(10 * np.log10(MSE_KF_data_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color= 'r', label = 'Extended Kalman Filter - data')

    plt.pyplot.title("Histogram [dB]")
    plt.pyplot.savefig('plt_hist_dB_0')


    KF_design_MSE_mean_dB = 10 * np.log10(np.mean(MSE_KF_design_linear_arr))
    KF_design_MSE_median_dB = 10 * np.log10(np.median(MSE_KF_design_linear_arr))
    KF_design_MSE_std_dB = 10 * np.log10(np.std(MSE_KF_design_linear_arr))
    print("Extended Kalman Filter - Design:",
          "MSE - mean", KF_design_MSE_mean_dB, "[dB]",
          "MSE - median", KF_design_MSE_median_dB, "[dB]",
          "MSE - std", KF_design_MSE_std_dB, "[dB]")

    KF_data_MSE_mean_dB = 10 * np.log10(np.mean(MSE_KF_data_linear_arr))
    KF_data_MSE_median_dB = 10 * np.log10(np.median(MSE_KF_data_linear_arr))
    KF_data_MSE_std_dB = 10 * np.log10(np.std(MSE_KF_data_linear_arr))
    print("Extended Kalman Filter - Data:",
          "MSE - mean", KF_data_MSE_mean_dB, "[dB]",
          "MSE - median", KF_data_MSE_median_dB, "[dB]",
          "MSE - std", KF_data_MSE_std_dB, "[dB]")

    KN_MSE_mean_dB = 10 * np.log10(np.mean(MSE_KN_linear_arr))
    KN_MSE_median_dB = 10 * np.log10(np.median(MSE_KN_linear_arr))
    KN_MSE_std_dB = 10 * np.log10(np.std(MSE_KN_linear_arr))

    print("kalman Net:",
          "MSE - mean", KN_MSE_mean_dB, "[dB]",
          "MSE - median", KN_MSE_median_dB, "[dB]",
          "MSE - std", KN_MSE_std_dB, "[dB]")

def KFPlot(res_grid):

    plt.pyplot.figure(figsize = (50, 20))
    x_plt = [-6, 0, 6]

    plt.pyplot.plot(x_plt, res_grid[0][:], 'xg', label='minus')
    plt.pyplot.plot(x_plt, res_grid[1][:], 'ob', label='base')
    plt.pyplot.plot(x_plt, res_grid[2][:], '+r', label='plus')
    plt.pyplot.plot(x_plt, res_grid[3][:], 'oy', label='base NN')

    plt.pyplot.legend()
    plt.pyplot.xlabel('Noise', fontsize=16)
    plt.pyplot.ylabel('MSE Loss Value [dB]', fontsize=16)
    plt.pyplot.title('Change', fontsize=16)
    plt.pyplot.savefig('plt_grid_dB')

    print("\ndistribution 1")
    print("Kalman Filter")
    print(res_grid[0][0], "[dB]", res_grid[1][0], "[dB]", res_grid[2][0], "[dB]")
    print(res_grid[1][0] - res_grid[0][0], "[dB]", res_grid[2][0] - res_grid[1][0], "[dB]")
    print("KalmanNet", res_grid[3][0], "[dB]", "KalmanNet Diff", res_grid[3][0] - res_grid[1][0], "[dB]")

    print("\ndistribution 2")
    print("Kalman Filter")
    print(res_grid[0][1], "[dB]", res_grid[1][1], "[dB]", res_grid[2][1], "[dB]")
    print(res_grid[1][1] - res_grid[0][1], "[dB]", res_grid[2][1] - res_grid[1][1], "[dB]")
    print("KalmanNet", res_grid[3][1], "[dB]", "KalmanNet Diff", res_grid[3][1] - res_grid[1][1], "[dB]")

    print("\ndistribution 3")
    print("Kalman Filter")
    print(res_grid[0][2], "[dB]", res_grid[1][2], "[dB]", res_grid[2][2], "[dB]")
    print(res_grid[1][2] - res_grid[0][2], "[dB]", res_grid[2][2] - res_grid[1][2], "[dB]")
    print("KalmanNet", res_grid[3][2], "[dB]", "KalmanNet Diff", res_grid[3][2] - res_grid[1][2], "[dB]")

def NNPlot_test(MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg,
           MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg):

    N_Epochs_plt = 100

    ###############################
    ### Plot per epoch [linear] ###
    ###############################
    plt.pyplot.figure(figsize = (50, 20))

    x_plt = range(0, N_Epochs_plt)

    # KNet - Test
    y_plt3 = MSE_test_linear_avg * np.ones(N_Epochs_plt)
    plt.pyplot.plot(x_plt, y_plt3, KColor[2], label=Klegend[2])

    # KF
    y_plt4 = MSE_KF_linear_avg * np.ones(N_Epochs_plt)
    plt.pyplot.plot(x_plt, y_plt4, KColor[3], label=Klegend[3])

    plt.pyplot.legend()
    plt.pyplot.xlabel('Number of Training Epochs', fontsize=16)
    plt.pyplot.ylabel('MSE Loss Value [linear]', fontsize=16)
    plt.pyplot.title('MSE Loss [linear] - per Epoch', fontsize=16)
    plt.pyplot.savefig('plt_model_test_linear')

    ###########################
    ### Plot per epoch [dB] ###
    ###########################
    plt.pyplot.figure(figsize = (50, 20))

    x_plt = range(0, N_Epochs_plt)

    # KNet - Test
    y_plt3 = MSE_test_dB_avg * np.ones(N_Epochs_plt)
    plt.pyplot.plot(x_plt, y_plt3, KColor[2], label=Klegend[2])

    # KF
    y_plt4 = MSE_KF_dB_avg * np.ones(N_Epochs_plt)
    plt.pyplot.plot(x_plt, y_plt4, KColor[3], label=Klegend[3])

    plt.pyplot.legend()
    plt.pyplot.xlabel('Number of Training Epochs', fontsize=16)
    plt.pyplot.ylabel('MSE Loss Value [dB]', fontsize=16)
    plt.pyplot.title('MSE Loss [dB] - per Epoch', fontsize=16)
    plt.pyplot.savefig('plt_model_test_dB')

    ########################
    ### Linear Histogram ###
    ########################
    plt.pyplot.figure(figsize=(50, 20))
    sns.distplot(MSE_test_linear_arr, hist=False, kde=True, kde_kws={'linewidth': 3}, color='g', label = 'KalmanNet')
    sns.distplot(MSE_KF_linear_arr, hist=False, kde=True, kde_kws={'linewidth': 3}, color= 'b', label = 'Kalman Filter')
    plt.pyplot.title("Histogram [Linear]")
    plt.pyplot.savefig('plt_hist_linear')

    fig, axes = plt.pyplot.subplots(2, 1, figsize=(50, 20), sharey=True, dpi=100)
    sns.distplot(MSE_test_linear_arr, hist=False, kde=True, kde_kws={'linewidth': 3}, color='g', label='KalmanNet', ax=axes[0])
    sns.distplot(MSE_KF_linear_arr, hist=False, kde=True, kde_kws={'linewidth': 3}, color='b', label='Kalman Filter', ax=axes[1])
    plt.pyplot.title("Histogram [Linear]")
    plt.pyplot.savefig('plt_hist_linear_1')

    ####################
    ### dB Histogram ###
    ####################

    plt.pyplot.figure(figsize=(50, 20))
    sns.distplot(10 * np.log10(MSE_test_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color='g', label = 'KalmanNet')
    sns.distplot(10 * np.log10(MSE_KF_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color= 'b', label = 'Kalman Filter')
    plt.pyplot.title("Histogram [dB]")
    plt.pyplot.savefig('plt_hist_dB')

    print('End')

def plotTrajectories(state_KNet, state_MKF, time, sensors, position, file_name,dpi=500):
    ## Plots the five states (results vs actual trajectory)
    # - Input: state [ax,ay,dy,vx,vy], time for x axis, position: {'MKF':[[x,y,head]xT],'KNet':[[x,y,head]xT]}, file_name (Folder/file_name.plot)
    # - Output: File with 6 plots 
    fig, axs = plt.pyplot.subplots(2, 3, figsize=(30,15))
    lw_sens = 0.3
    lw_imp = 0.9
    fs = 'x-large'

    state_KNet = state_KNet.detach()
    state_MKF = state_MKF.detach()
    sensors = sensors.detach()

    if (time is None):
        time = torch.linspace(0, state_KNet.size(1), steps=state_KNet.size(1))

    state_KNet = state_KNet.to(torch.device("cpu"),non_blocking=True)
    state_MKF = state_MKF.to(torch.device("cpu"),non_blocking=True)
    time = time.to(torch.device("cpu"),non_blocking=True)
    sensors = sensors.to(torch.device("cpu"),non_blocking=True)
    if position is not None:
      position['KNet'] = position['KNet'].to(torch.device("cpu"),non_blocking=True)
      position['MKF'] = position['MKF'].to(torch.device("cpu"),non_blocking=True)
      position['RTK'] = position['RTK'].to(torch.device("cpu"),non_blocking=True)

    # a_x
    axs[0, 0].plot(time, state_KNet[0,:], 'b', linewidth=lw_imp)
    axs[0, 0].plot(time, state_MKF[0,:], 'r', linewidth=lw_imp)
    axs[0, 0].plot(time, sensors[0,:], 'g', linewidth=lw_sens) # IMU
    axs[0, 0].plot(time, sensors[3,:], 'y', linewidth=lw_sens) # INS
    axs[0, 0].set_title("a_x(t)")
    axs[0,0].legend(('KNet', 'GT(MKF)','IMU', 'INS'), fontsize=fs)

    # a_y
    axs[0, 1].plot(time, state_KNet[1,:], 'b', linewidth=lw_imp)
    axs[0, 1].plot(time, state_MKF[1,:], 'r', linewidth=lw_imp)
    axs[0, 1].plot(time, sensors[1,:], 'g', linewidth=lw_sens) # IMU
    axs[0, 1].plot(time, sensors[4,:], 'y', linewidth=lw_sens) # INS
    axs[0, 1].set_title("a_y(t)")
    axs[0,1].legend(('KNet', 'GT(MKF)','IMU', 'INS'), fontsize=fs)

    # dyaw
    axs[0, 2].plot(time, state_KNet[2,:], 'b', linewidth=lw_imp)
    axs[0, 2].plot(time, state_MKF[2,:], 'r', linewidth=lw_imp)
    axs[0, 2].plot(time, sensors[2,:], 'g', linewidth=lw_sens) # IMU
    axs[0, 2].plot(time, sensors[5,:], 'y', linewidth=lw_sens) # INS
    axs[0, 2].plot(time, sensors[6,:], 'm', linewidth=lw_sens) # WS
    axs[0, 2].set_title("yaw rate(t)")
    axs[0,2].legend(('KNet', 'GT(MKF)','IMU', 'INS','WS'), fontsize=fs)

    # v_x
    axs[1, 0].plot(time, state_KNet[3,:], 'b', linewidth=lw_imp)
    axs[1, 0].plot(time, state_MKF[3,:], 'r', linewidth=lw_imp)
    axs[1, 0].plot(time, sensors[7,:], 'm', linewidth=lw_sens)
    #axs[1, 0].plot(time, sensors[3,:], 'm', linewidth=lw_sens) # WS
    axs[1, 0].set_title("v_x(t)")
    axs[1,0].legend(('KNet', 'GT(MKF)','WS'), fontsize=fs)

    # v_y
    axs[1, 1].plot(time, state_KNet[4,:], 'b', linewidth=lw_imp)
    axs[1, 1].plot(time, state_MKF[4,:], 'r', linewidth=lw_imp)
    #axs[1, 1].plot(time, sensors[4,:], 'm', linewidth=lw_sens) # WS
    axs[1, 1].plot(time, sensors[8,:], 'm', linewidth=lw_sens)
    axs[1, 1].set_title("v_y(t)")
    axs[1,1].legend(('KNet', 'GT(MKF)','WS'), fontsize=fs)

    # Velocity integration
    if position is not None:
        axs[1,2].plot(position['KNet'][0,:],position['KNet'][1,:],'b',linewidth=lw_imp)
        axs[1,2].plot(position['MKF'][0,:],position['MKF'][1,:],'r',linewidth=lw_imp)
        axs[1,2].plot(position['RTK'][0,:],position['RTK'][1,:],'g',linewidth=lw_sens)
        axs[1,2].set_title("Integrated velocity (x,y)")
        axs[1,2].legend(('KNet', 'GT(MKF)','RTK'), fontsize=fs)

    fig.tight_layout()
    plt.pyplot.savefig(file_name,dpi=dpi)
    print("Trajectory plot successfuly saved")
    plt.pyplot.close('all')
