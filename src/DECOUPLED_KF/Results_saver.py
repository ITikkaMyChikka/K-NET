from datetime import datetime
import matplotlib.pyplot as plt
import torch
from Encoder import integrate_vel, geodetic_transform, velocity_integration

def generate_results_file(results_path, KGain, q, MSE, KGain_y, q_opt_y, MSE_y):

    file = open(results_path+"results.txt","w")
    # Title
    file.write("TEST RESULTS FILE\n")

    # Results
    file.write("vx MSE:"    + str(MSE[0].item()) +"[dB], "+
               "\nax MSE:"  + str(MSE[1].item()) +"[dB], "+
               "\nvy MSE:"  + str(MSE_y[0].item()) +"[dB], "+
               "\nay MSE:"  + str(MSE_y[1].item()) +"[dB], ")

    file.write("\n\nOptimal q_value (x-axis):"+ str(q))
    file.write("\n\nOptimal q_value (y-axis):" + str(q_opt_y))
    file.write("\n\nKalman Gain (x-axis):" + str(KGain))
    file.write("\n\nKalman Gain (y-axis):" + str(KGain_y))

    # Date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    file.write("\n\n\n\nSimulation Date & Time:"+ dt_string+"\n\n\n")

    file.close()
    print("results.txt successfully saved.")


def plot_stuff(plots_path, x_opt, y_opt, test_target, test_time, T_test, test_input, rtk_test):

    #######################
    ### TRAJECTORY PLOT ###
    ### vx,vy,ax,ay,pos ###
    #######################

    # Plot the states (vx, ax) & (vy,ay) and trajectory (px,py) and save it in trajectories.png
    file_name = plots_path + 'trajectories.png'
    fig, axs = plt.subplots(2, 3, figsize=(30, 15))
    lw_sens = 0.3
    lw_imp = 0.9
    fs = 'x-large'
    dpi = 500
    plot_limit = round(T_test)
    plot_start = round(0)

    state_KF = x_opt[0, :, plot_start:plot_limit]
    state_KF_y = y_opt[0, :, plot_start:plot_limit]
    state_MKF = test_target.detach()[0, :, plot_start:plot_limit]
    time = test_time.detach()[0, 0, plot_start:plot_limit]
    # sensors = z_full.detach()[0,:,plot_start:plot_limit]

    # a_x
    axs[0, 0].plot(time, state_KF[1, :], 'b', linewidth=lw_imp)
    axs[0, 0].plot(time, state_MKF[0, :], 'r', linewidth=lw_imp)
    axs[0, 0].set_title("a_x(t)")
    axs[0, 0].legend(('Kalman Filter', 'GT(MKF)'), fontsize=fs)

    # v_x
    axs[0, 1].plot(time, state_KF[0, :], 'b', linewidth=lw_imp)
    axs[0, 1].plot(time, state_MKF[3, :], 'r', linewidth=lw_imp)
    axs[0, 1].set_title("v_x(t)")
    axs[0, 1].legend(('Kalman Filter', 'GT(MKF)'), fontsize=fs)

    # a_y
    axs[1, 0].plot(time, state_KF_y[1, :], 'b', linewidth=lw_imp)
    axs[1, 0].plot(time, state_MKF[1, :], 'r', linewidth=lw_imp)
    axs[1, 0].set_title("a_y(t)")
    axs[1, 0].legend(('Kalman Filter', 'GT(MKF)'), fontsize=fs)

    # v_y
    axs[1, 1].plot(time, state_KF_y[0, :], 'b', linewidth=lw_imp)
    axs[1, 1].plot(time, state_MKF[4, :], 'r', linewidth=lw_imp)
    axs[1, 1].set_title("v_y(t)")
    axs[1, 1].legend(('Kalman Filter', 'GT(MKF)'), fontsize=fs)

    # positions (px,py) from KF
    pos_KF = integrate_vel(x_opt, y_opt, test_time[0,0,:], test_input, T_test)

    # position from MKF (GT)
    pos_MKF = velocity_integration(test_target[0, :, :].detach(), test_time[0, 0, :])

    # Transform RTK to ENU location
    lat0 = torch.mean(rtk_test[0, 1, 0:10])
    lon0 = torch.mean(rtk_test[0, 0, 0:10])
    pos_RTK = geodetic_transform(rtk_test[0, :, :], lat0, lon0)

    position = {'KF': pos_KF.detach()[:, plot_start:plot_limit],
                      'MKF': pos_MKF.detach()[:, plot_start:plot_limit],
                      'RTK': pos_RTK.detach()[:, plot_start:plot_limit]}

    # Velocity integration
    if position is not None:
        axs[0, 2].plot(position['KF'][0, :], position['KF'][1, :], 'b', linewidth=lw_imp)
        axs[0, 2].plot(position['MKF'][0, :], position['MKF'][1, :], 'r', linewidth=lw_imp)
        axs[0, 2].plot(position['RTK'][0, :], position['RTK'][1, :], 'g', linewidth=lw_sens)
        axs[0, 2].set_title("Integrated velocity (x,y)")
        axs[0, 2].legend(('KF', 'GT(MKF)', 'RTK'), fontsize=fs)

    fig.tight_layout()
    plt.savefig(file_name, dpi=dpi)
    print("Trajectory plot successfully saved")
    plt.close('all')

    ###################
    ### ERROR PLOT  ###
    ### vx,vy,ax,ay ###
    ###################

    # Plot Error Graphs of (vx,ax) and (vy,ay)
    file_name = plots_path + 'error_graphs.png'
    fig, axs = plt.subplots(2, 2, figsize=(30, 15))
    lw_sens = 0.3
    lw_imp = 0.9
    fs = 'x-large'
    dpi = 500

    # Error Graph a_x ((KF(t) – GR(t))^2 in [dB] (10log))
    error_ax = torch.pow((state_KF[1, :] - state_MKF[0, :]), 2)  # (ax - ax_GT)^2
    error_ax_dB = 10 * torch.log10(error_ax)
    axs[0, 0].plot(time, error_ax_dB, 'r', linewidth=lw_imp)
    axs[0, 0].set_title("Error a_x in dB")

    # Error Graph v_x
    error_vx = torch.pow((state_KF[0, :] - state_MKF[3, :]), 2)  # (vx - vx_GT)^2
    error_vx_dB = 10 * torch.log10(error_vx)
    axs[0, 1].plot(time, error_vx_dB, 'r', linewidth=lw_imp)
    axs[0, 1].set_title("Error v_x in dB")

    # Error Graph a_y
    error_ay = torch.pow((state_KF_y[1, :] - state_MKF[1, :]), 2)  # (ax - ax_GT)^2
    error_ay_dB = 10 * torch.log10(error_ay)
    axs[1, 0].plot(time, error_ay_dB, 'r', linewidth=lw_imp)
    axs[1, 0].set_title("Error a_y in dB")

    # Error Graph v_y
    error_vy = torch.pow((state_KF_y[0, :] - state_MKF[4, :]), 2)  # (ax - ax_GT)^2
    error_vy_dB = 10 * torch.log10(error_vy)
    axs[1, 1].plot(time, error_vy_dB, 'r', linewidth=lw_imp)
    axs[1, 1].set_title("Error v_y in dB")

    fig.tight_layout()
    plt.savefig(file_name, dpi=dpi)
    print("Error plot successfully saved")
    plt.close('all')
