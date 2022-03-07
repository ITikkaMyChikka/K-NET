import torch.nn as nn
import torch
import time
from tqdm import tqdm
import numpy as np
import src.parameters as params


class KalmanFilter:

    def __init__(self, SystemModel):
        self.F = SystemModel.F
        self.m = SystemModel.m

        # Has to be transformed because of EKF non-linearity
        self.Q = SystemModel.Q

        self.H = SystemModel.H
        self.n = SystemModel.n

        # Has to be transformed because of EKF non-linearity
        self.R = SystemModel.R

        self.T = SystemModel.T

        # Pre allocate KG array
        self.KG_array = torch.zeros((self.T, self.m, self.n))


    # Predict
    def Predict(self):
        # Predict the 1-st moment of x
        self.m1x_prior = torch.squeeze(torch.matmul(self.F, self.m1x_posterior))

        # Adding noise to the velocity
        #self.m1x_prior[1] = self.m1x_prior[1]+np.random.normal(torch.zeros(1), params.q_cv, 1)

        # Predict the 2-nd moment of x
        self.m2x_prior = torch.matmul(self.F, self.m2x_posterior)
        self.m2x_prior = torch.matmul(self.m2x_prior, self.F.T) + self.Q

        # Predict the 1-st moment of y
        self.m1y = torch.squeeze(torch.matmul(self.H, self.m1x_prior))
        # Predict the 2-nd moment of y
        self.m2y = torch.matmul(self.H, self.m2x_prior)
        self.m2y = torch.matmul(self.m2y, self.H.T) + self.R

    # Compute the Kalman Gain
    def KGain(self):

        self.KG = torch.matmul(self.m2x_prior, self.H.T)
        self.KG = torch.matmul(self.KG, torch.inverse(self.m2y))

        # Save KalmanGain
        self.KG_array[self.i] = self.KG
        self.i += 1

    # Innovation
    def Innovation(self, y):
        self.dy = y - self.m1y

    # Compute Posterior
    def Correct(self):
        # Compute the 1-st posterior moment
        self.m1x_posterior = self.m1x_prior + torch.matmul(self.KG, self.dy)

        # Compute the 2-nd posterior moment
        self.m2x_posterior = torch.matmul(self.m2y, torch.transpose(self.KG, 0, 1))
        self.m2x_posterior = self.m2x_prior - torch.matmul(self.KG, self.m2x_posterior)

    def Update(self, y):
        self.Predict()
        self.KGain()
        self.Innovation(y)
        self.Correct()

        return self.m1x_posterior, self.m2x_posterior

    def Update_Error(self, y):
        self.Predict()
        self.KGain()
        self.Innovation(y)
        self.Correct()

        return self.m1x_posterior, self.m2x_posterior

    def InitSequence(self, m1x_0, m2x_0):
        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0

    ######################################
    ### Generate interpolated Sequence ###
    ######################################
    def GenerateSequence_inter(self, y, T):
        # Pre allocate an array for predicted state and variance
        self.x = torch.empty(size=[self.m, T])
        self.sigma = torch.empty(size=[self.m, self.m, T])
        # Pre allocate KG array
        self.KG_array = torch.zeros((T, self.m, self.n))
        self.i = 0  # Index for KG_array alocation

        self.m1x_posterior = torch.squeeze(self.m1x_0)
        self.m2x_posterior = torch.squeeze(self.m2x_0)

        for t in range(0, T):
            # Only Prediction Steps
            self.Predict()
            self.KGain()
            # Save the prediction
            xt, sigmat = self.m1x_prior, self.m2x_prior

            # If we have an observation
            if t%params.ratio_cv == (params.ratio_cv-1):
                # We perform update step
                t_low = int((t+1)/params.ratio_cv - 1)
                yt = torch.squeeze(y[:, t_low])  # Every 10th step we take a new observation
                self.Innovation(yt)
                self.Correct()
                # Save the updated prediction
                xt, sigmat = self.m1x_posterior, self.m2x_posterior
            else:
                # In case we don't perform an update we say that posterior = prior
                self.m1x_posterior = xt
                self.m2x_posterior = sigmat

            # Save eihter the prediction or updated prediction to our output
            self.x[:, t] = torch.squeeze(xt)
            self.sigma[:, :, t] = torch.squeeze(sigmat)

    #########################
    ### Generate Sequence ###
    #########################

    def GenerateSequence(self, y, T):
        # Pre allocate an array for predicted state and variance
        self.x = torch.empty(size=[self.m, T])
        self.sigma = torch.empty(size=[self.m, self.m, T])
        # Pre allocate KG array
        self.KG_array = torch.zeros((T, self.m, self.n))
        self.i = 0  # Index for KG_array alocation

        self.m1x_posterior = torch.squeeze(self.m1x_0)
        self.m2x_posterior = torch.squeeze(self.m2x_0)

        for t in range(0, T):
            # Extract observation at time step t
            yt = torch.squeeze(y[:, t])
            # Perform prediction and update step
            xt, sigmat = self.Update(yt)
            # Save the posterior results
            self.x[:, t] = torch.squeeze(xt)
            self.sigma[:, :, t] = torch.squeeze(sigmat)

    ########################################
    ### Generate Sequence Error State KF ###
    ########################################

    def GenerateSequence_ERROR(self, y, T):
        # Pre allocate an array for predicted state and variance
        self.x = torch.empty(size=[self.m, T])
        self.sigma = torch.empty(size=[self.m, self.m, T])
        # Pre allocate KG array
        self.KG_array = torch.zeros((T, self.m, self.n))
        self.i = 0  # Index for KG_array alocation

        self.m1x_posterior = torch.squeeze(self.m1x_0)
        self.m2x_posterior = torch.squeeze(self.m2x_0)

        for t in range(0, T):
            # Extract observation at time step t
            yt = torch.squeeze(y[:, t])
            # Perform prediction and update step
            xt, sigmat = self.Update_Error(yt)
            # Save the posterior results
            self.x[:, t] = torch.squeeze(xt)
            self.sigma[:, :, t] = torch.squeeze(sigmat)



def KF_inter_Test(SysModel, input, target):
    N_T = target.size()[0]  # number of samples

    # LOSS
    loss_fn = nn.MSELoss(reduction='mean')

    # MSE [Linear]
    MSE_EKF_linear_arr = torch.empty(N_T)
    MSE_state = torch.zeros((N_T, target.size()[1]))

    KF = KalmanFilter(SysModel)
    KF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)

    KG_array = torch.zeros_like(KF.KG_array)
    KF_out = torch.empty([N_T, SysModel.m, SysModel.T])
    start = time.time()
    for j in tqdm(range(0, N_T)):
        #print("Filtering sample number: ", j)
        KF.GenerateSequence_inter(input[j, :, :], KF.T)
        MSE_EKF_linear_arr[j] = loss_fn(KF.x, target[j, :, :]).item()

        for state in range(0, target.size()[1]):
            MSE_state[j, state] = loss_fn(KF.x[state], target[j, state, :]).item()

        KG_array = torch.add(KF.KG_array, KG_array)
        KF_out[j, :, :] = KF.x
    end = time.time()
    t = end - start
    # Average KG_array over Test Examples
    KG_array /= N_T

    MSE_state_linear_avg = torch.mean(MSE_state, dim=0)
    MSE_state_dB_avg = 10 * torch.log10(MSE_state_linear_avg)

    MSE_EKF_linear_avg = torch.mean(MSE_EKF_linear_arr)
    MSE_EKF_dB_avg = 10 * torch.log10(MSE_EKF_linear_avg)

    # Standard deviation
    MSE_EKF_dB_std = torch.std(MSE_EKF_linear_arr, unbiased=True)
    MSE_EKF_dB_std = 10 * torch.log10(MSE_EKF_dB_std)

    print("KF - MSE LOSS:", MSE_EKF_dB_avg, "[dB]")
    print("KF - MSE STD:", MSE_EKF_dB_std, "[dB]")
    # Print Run Time
    print("Inference Time:", t)

    return [MSE_state_dB_avg, MSE_EKF_dB_avg, KG_array, KF_out]

def KF_Test(SysModel, input, target):
    N_T = target.size()[0]  # number of samples

    # LOSS
    loss_fn = nn.MSELoss(reduction='mean')

    # MSE [Linear]
    MSE_EKF_linear_arr = torch.empty(N_T)
    MSE_EKF_linear_arr_pos = torch.empty(N_T)
    MSE_EKF_linear_arr_vel = torch.empty(N_T)

    KF_I = KalmanFilter(SysModel)
    KF_I.InitSequence(SysModel.m1x_0, SysModel.m2x_0)

    KG_array = torch.zeros_like(KF_I.KG_array)
    KF_out = torch.empty([N_T, SysModel.m, SysModel.T])
    start = time.time()
    for j in tqdm(range(0, N_T)):
        #print("Filtering sample number: ", j)
        KF_I.GenerateSequence(input[j, :, :], KF_I.T)

        # TOTAL MSE LOSS
        MSE_EKF_linear_arr[j] = loss_fn(KF_I.x, target[j, :, :]).item()
        # POS MSE LOSS
        MSE_EKF_linear_arr_pos[j] = loss_fn(KF_I.x[0:3:2, :], target[j, 0:3:2, :])
        # VEL MSE LOSS
        MSE_EKF_linear_arr_vel[j] = loss_fn(KF_I.x[1:4:2, :], target[j, 1:4:2, :])



        KG_array = torch.add(KF_I.KG_array, KG_array)
        KF_out[j, :, :] = KF_I.x
    end = time.time()
    t = end - start
    # Average KG_array over Test Examples
    KG_array /= N_T

    #  (px_est - px_gt)^2 + (py_est - py_gt)^2 / 2
    MSE_EKF_linear_avg_pos = torch.mean(MSE_EKF_linear_arr_pos)
    MSE_EKF_dB_avg_pos = 10 * torch.log10(MSE_EKF_linear_avg_pos)

    #  (vx_est - vx_gt)^2 + (vy_est - vy_gt)^2 / 2
    MSE_EKF_linear_avg_vel = torch.mean(MSE_EKF_linear_arr_vel)
    MSE_EKF_dB_avg_vel = 10 * torch.log10(MSE_EKF_linear_avg_vel)

    # (px_est - px_gt)^2 + (py_est - py_gt)^2 + (vx_est - vx_gt)^2 + (vy_est - vy_gt)^2 / 4
    MSE_EKF_linear_avg = torch.mean(MSE_EKF_linear_arr)
    MSE_EKF_dB_avg = 10 * torch.log10(MSE_EKF_linear_avg)

    # Standard deviation
    MSE_EKF_dB_std = torch.std(MSE_EKF_linear_arr, unbiased=True)
    MSE_EKF_dB_std = 10 * torch.log10(MSE_EKF_dB_std)

    print("KF - MSE LOSS:", MSE_EKF_dB_avg, "[dB]")
    print("KF - MSE STD:", MSE_EKF_dB_std, "[dB]")
    # Print Run Time
    print("Inference Time:", t)

    return [MSE_EKF_dB_avg, MSE_EKF_dB_avg_pos, MSE_EKF_dB_avg_vel, KF_out]


def KF_Test1(SysModel, input, target):
    N_T = target.size()[0]  # number of samples

    # LOSS
    loss_fn = nn.MSELoss(reduction='mean')

    # MSE [Linear]
    MSE_EKF_linear_arr = torch.empty(N_T)
    MSE_EKF_linear_arr_pos = torch.empty(N_T)
    MSE_EKF_linear_arr_vel = torch.empty(N_T)
    MSE_EKF_linear_arr_acc = torch.empty(N_T)

    KF_I = KalmanFilter(SysModel)
    KF_I.InitSequence(SysModel.m1x_0, SysModel.m2x_0)

    KG_array = torch.zeros_like(KF_I.KG_array)
    KF_out = torch.empty([N_T, SysModel.m, SysModel.T])
    start = time.time()
    for j in tqdm(range(0, N_T)):
        #print("Filtering sample number: ", j)
        KF_I.GenerateSequence(input[j, :, :], KF_I.T)

        # TOTAL MSE LOSS
        MSE_EKF_linear_arr[j] = loss_fn(KF_I.x, target[j, :, :]).item()
        # POS MSE LOSS
        MSE_EKF_linear_arr_pos[j] = loss_fn(KF_I.x[0:6:3, :], target[j, 0:6:3, :])
        # VEL MSE LOSS
        MSE_EKF_linear_arr_vel[j] = loss_fn(KF_I.x[1:6:3, :], target[j, 1:6:3, :])
        # ACC MSE LOSS
        MSE_EKF_linear_arr_acc[j] = loss_fn(KF_I.x[2:6:3, :], target[j, 2:6:3, :])


        KG_array = torch.add(KF_I.KG_array, KG_array)
        KF_out[j, :, :] = KF_I.x
    end = time.time()
    t = end - start
    # Average KG_array over Test Examples
    KG_array /= N_T

    #  (px_est - px_gt)^2 + (py_est - py_gt)^2 / 2
    MSE_EKF_linear_avg_pos = torch.mean(MSE_EKF_linear_arr_pos)
    MSE_EKF_dB_avg_pos = 10 * torch.log10(MSE_EKF_linear_avg_pos)

    #  (vx_est - vx_gt)^2 + (vy_est - vy_gt)^2 / 2
    MSE_EKF_linear_avg_vel = torch.mean(MSE_EKF_linear_arr_vel)
    MSE_EKF_dB_avg_vel = 10 * torch.log10(MSE_EKF_linear_avg_vel)

    #  (ax_est - ax_gt)^2 + (ay_est - ay_gt)^2 / 2
    MSE_EKF_linear_avg_acc = torch.mean(MSE_EKF_linear_arr_acc)
    MSE_EKF_dB_avg_acc = 10 * torch.log10(MSE_EKF_linear_avg_acc)

    # (px_est - px_gt)^2 + (py_est - py_gt)^2 + (vx_est - vx_gt)^2 + (vy_est - vy_gt)^2 / 4
    MSE_EKF_linear_avg = torch.mean(MSE_EKF_linear_arr)
    MSE_EKF_dB_avg = 10 * torch.log10(MSE_EKF_linear_avg)

    # Standard deviation
    MSE_EKF_dB_std = torch.std(MSE_EKF_linear_arr, unbiased=True)
    MSE_EKF_dB_std = 10 * torch.log10(MSE_EKF_dB_std)

    print("KF - MSE LOSS:", MSE_EKF_dB_avg, "[dB]")
    print("KF - MSE STD:", MSE_EKF_dB_std, "[dB]")

    # Print Run Time
    print("Inference Time:", t)

    return [MSE_EKF_dB_avg, MSE_EKF_dB_avg_pos, MSE_EKF_dB_avg_vel, MSE_EKF_dB_avg_acc, KF_out]
















class ExtendedKalmanFilter:

    def __init__(self, SystemModel):
        self.F = SystemModel.F
        self.F_Jacobian = SystemModel.F_Jacobian
        self.m = SystemModel.m

        # Has to be transformed because of EKF non-linearity
        self.Q = SystemModel.Q

        self.H = SystemModel.H
        self.n = SystemModel.n

        # Has to be transformed because of EKF non-linearity
        self.R = SystemModel.R

        self.T = SystemModel.T

        # Pre allocate KG array
        #self.KG_array = torch.zeros((self.T, self.m, self.n))


    # Predict
    def Predict(self):
        # Predict the 1-st moment of x
        self.m1x_prior = torch.squeeze(self.F(self.m1x_posterior))
        #self.m1x_prior = torch.squeeze(torch.matmul(self.F_Jacobian(self.m1x_posterior), self.m1x_posterior))
        #self.m1x_prior = torch.squeeze(torch.matmul(self.F, self.m1x_posterior))

        # Adding noise to the velocity
        #self.m1x_prior[1] = self.m1x_prior[1]+np.random.normal(torch.zeros(1), params.q_cv, 1)

        # Predict the 2-nd moment of x
        # TODO JACOBIAN
        self.m2x_prior = torch.matmul(self.F_Jacobian(self.m1x_prior), self.m2x_posterior)
        #self.m2x_prior = torch.matmul(self.F, self.m2x_posterior)
        self.m2x_prior = torch.matmul(self.m2x_prior, self.F_Jacobian(self.m1x_prior).T) + self.Q

        # Predict the 1-st moment of y
        self.m1y = torch.squeeze(torch.matmul(self.H(self.m1x_prior), self.m1x_prior))
        # Predict the 2-nd moment of y
        self.m2y = torch.matmul(self.H(self.m1x_prior), self.m2x_prior)
        self.m2y = torch.matmul(self.m2y, self.H(self.m1x_prior).T) + self.R(self.m1x_prior)


    # Compute the Kalman Gain
    def KGain(self):

        self.KG = torch.matmul(self.m2x_prior, self.H(self.m1x_prior).T)
        try:
            inverse = torch.inverse(self.m2y)
        except:
            print("self.Q: ", self.Q)
            print("m2x_prior: ", self.m2x_prior)
            print("m2y: ", self.m2y)
            print("noise Matrix: ", self.R(self.m1x_prior))
            print("m2y + noise: ", self.m2y + self.R(self.m1x_prior))
            m2y = self.m2y
            print(torch.linalg.matrix_rank(m2y))
            inverse = torch.inverse(m2y)
        self.KG = torch.matmul(self.KG, inverse)

        # Save KalmanGain
        #self.KG_array[self.i] = self.KG
        self.i += 1

    # Innovation
    def Innovation(self, y):
        self.dy = y - self.m1y

    # Compute Posterior
    def Correct(self):
        # Compute the 1-st posterior moment
        self.m1x_posterior = self.m1x_prior + torch.matmul(self.KG, self.dy)

        # Compute the 2-nd posterior moment
        self.m2x_posterior = torch.matmul(self.m2y, torch.transpose(self.KG, 0, 1))
        self.m2x_posterior = self.m2x_prior - torch.matmul(self.KG, self.m2x_posterior)

    def Update(self, y):
        self.Predict()
        self.KGain()
        self.Innovation(y)
        self.Correct()

        return self.m1x_posterior, self.m2x_posterior

    def Update_Error(self, y):
        self.Predict()
        self.KGain()
        self.Innovation(y)
        self.Correct()

        return self.m1x_posterior, self.m2x_posterior

    def InitSequence(self, m1x_0, m2x_0):
        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0


    #########################
    ### Generate Sequence ###
    #########################

    def GenerateSequence(self, y, T):
        # Pre allocate an array for predicted state and variance
        self.x = torch.empty(size=[self.m, T])
        self.sigma = torch.empty(size=[self.m, self.m, T])
        # Pre allocate KG array
        #self.KG_array = torch.zeros((T, self.m, self.n))
        self.i = 0  # Index for KG_array alocation

        self.m1x_posterior = torch.squeeze(self.m1x_0)
        self.m2x_posterior = torch.squeeze(self.m2x_0)

        for t in range(0, T):
            # Extract observation at time step t
            yt = torch.squeeze(y[:, t])
            # Perform prediction and update step
            xt, sigmat = self.Update(yt)
            # Save the posterior results
            self.x[:, t] = torch.squeeze(xt)
            self.sigma[:, :, t] = torch.squeeze(sigmat)



def KF_Test2(SysModel, input, target):
    N_T = target.size()[0]  # number of samples

    # LOSS
    loss_fn = nn.MSELoss(reduction='mean')

    # MSE [Linear]
    MSE_EKF_linear_arr = torch.empty(N_T)
    MSE_EKF_linear_arr_pos = torch.empty(N_T)
    MSE_EKF_linear_arr_vel = torch.empty(N_T)
    MSE_EKF_linear_arr_acc = torch.empty(N_T)

    EKF = ExtendedKalmanFilter(SysModel)
    EKF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)

    #KG_array = torch.zeros_like(EKF.KG_array)
    KF_out = torch.empty([N_T, SysModel.m, SysModel.T])
    start = time.time()
    for j in tqdm(range(0, N_T)):
        #print("Filtering sample number: ", j)
        EKF.GenerateSequence(input[j, :, :], EKF.T)

        # TOTAL MSE LOSS
        MSE_EKF_linear_arr[j] = loss_fn(EKF.x, target[j, :, :]).item()
        # POS MSE LOSS
        MSE_EKF_linear_arr_pos[j] = loss_fn(EKF.x[0:2, :], target[j, 0:2, :])
        # VEL MSE LOSS
        MSE_EKF_linear_arr_vel[j] = loss_fn(EKF.x[3, :], target[j, 3, :])
        # ACC MSE LOSS
        MSE_EKF_linear_arr_acc[j] = loss_fn(EKF.x[4:6, :], target[j, 4:6, :])


        #KG_array = torch.add(EKF.KG_array, KG_array)
        KF_out[j, :, :] = EKF.x
    end = time.time()
    t = end - start
    # Average KG_array over Test Examples
    #KG_array /= N_T

    #  (px_est - px_gt)^2 + (py_est - py_gt)^2 / 2
    MSE_EKF_linear_avg_pos = torch.mean(MSE_EKF_linear_arr_pos)
    MSE_EKF_dB_avg_pos = 10 * torch.log10(MSE_EKF_linear_avg_pos)

    #  (vx_est - vx_gt)^2 + (vy_est - vy_gt)^2 / 2
    MSE_EKF_linear_avg_vel = torch.mean(MSE_EKF_linear_arr_vel)
    MSE_EKF_dB_avg_vel = 10 * torch.log10(MSE_EKF_linear_avg_vel)

    #  (ax_est - ax_gt)^2 + (ay_est - ay_gt)^2 / 2
    MSE_EKF_linear_avg_acc = torch.mean(MSE_EKF_linear_arr_acc)
    MSE_EKF_dB_avg_acc = 10 * torch.log10(MSE_EKF_linear_avg_acc)

    # (px_est - px_gt)^2 + (py_est - py_gt)^2 + (vx_est - vx_gt)^2 + (vy_est - vy_gt)^2 / 4
    MSE_EKF_linear_avg = torch.mean(MSE_EKF_linear_arr)
    MSE_EKF_dB_avg = 10 * torch.log10(MSE_EKF_linear_avg)

    # Standard deviation
    MSE_EKF_dB_std = torch.std(MSE_EKF_linear_arr, unbiased=True)
    MSE_EKF_dB_std = 10 * torch.log10(MSE_EKF_dB_std)

    print("KF - MSE LOSS:", MSE_EKF_dB_avg, "[dB]")
    print("KF - MSE STD:", MSE_EKF_dB_std, "[dB]")

    # Print Run Time
    print("Inference Time:", t)

    return [MSE_EKF_dB_avg, MSE_EKF_dB_avg_pos, MSE_EKF_dB_avg_vel, MSE_EKF_dB_avg_acc, KF_out]






























###############################
### KALMAN FILTER FOR KITTI ###
###############################



def KF_Test_KITTI(SysModel, input_pos, input_acc, target, GT_time, time_list, time_index_list):

    # LOSS
    loss_fn = nn.MSELoss(reduction='mean')

    # MSE [Linear]
    MSE_EKF_linear_arr = 0
    MSE_KF_arr = []
    MSE_EKF_linear_arr_pos = 0

    KF = KalmanFilter_KITTI(SysModel)
    KF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)

    #KG_array = torch.zeros_like(KF.KG_array)
    KF_out = torch.empty([SysModel.m, SysModel.T])
    start = time.time()


    # RUN KF
    KF.GenerateSequence(input_pos, input_acc, KF.T, time_list, time_index_list)

    # Compute Loss
    gt_index = 0
    gttime = GT_time[-1]-GT_time[0]
    for t in range(0, gttime):
        if t == GT_time[gt_index]:
            KF_pos = torch.zeros(2)
            KF_pos[0] = KF.x[0, t]
            KF_pos[1] = KF.x[3, t]
            loss = loss_fn(KF_pos, target[gt_index, :]).item()
            MSE_KF_arr.append(loss)
            gt_index += 1
    print("Size of loss array: ", len(MSE_KF_arr))
    print("Size of GT array: ", target.size())
    print("Size of GT time array: ", len(GT_time))
    print("How man gt samples we use: ", gt_index)


    #MSE_EKF_linear_arr = loss_fn(KF.x, target[:, :]).item()

    #KG_array = torch.add(KF.KG_array, KG_array)
    KF_out[:, :] = KF.x

    end = time.time()
    t = end - start

    MSE_EKF_linear_arr = torch.FloatTensor(MSE_KF_arr)
    # (px_est - px_gt)^2 + (py_est - py_gt)^2 + (vx_est - vx_gt)^2 + (vy_est - vy_gt)^2 / 4
    MSE_EKF_linear_avg = torch.mean(MSE_EKF_linear_arr)
    MSE_EKF_dB_avg = 10 * torch.log10(MSE_EKF_linear_avg)

    print("KF - MSE LOSS:", MSE_EKF_dB_avg, "[dB]")
    # Print Run Time
    print("Inference Time:", t)

    return [MSE_EKF_dB_avg, KF_out]




class KalmanFilter_KITTI:

    def __init__(self, SystemModel):
        self.F = SystemModel.F
        self.m = SystemModel.m

        # Has to be transformed because of EKF non-linearity
        self.Q = SystemModel.Q

        self.H_PO = SystemModel.H_PO
        self.H_AO = SystemModel.H_AO
        self.H_PAO = SystemModel.H_PAO
        self.n = SystemModel.n

        # Has to be transformed because of EKF non-linearity
        self.R_PO = SystemModel.R_PO
        self.R_AO = SystemModel.R_AO
        self.R_PAO = SystemModel.R_PAO

        self.T = SystemModel.T

        # Pre allocate KG array
        #self.KG_array = torch.zeros((self.T, self.m, self.n))


    # Predict
    def Predict(self, obs_model):
        # Predict the 1-st moment of x
        self.m1x_prior = torch.squeeze(torch.matmul(self.F, self.m1x_posterior))

        # Adding noise to the velocity
        #self.m1x_prior[1] = self.m1x_prior[1]+np.random.normal(torch.zeros(1), params.q_cv, 1)

        # Predict the 2-nd moment of x
        self.m2x_prior = torch.matmul(self.F, self.m2x_posterior)
        self.m2x_prior = torch.matmul(self.m2x_prior, self.F.T) + self.Q

        # Predict the 1-st moment of y
        if obs_model == 0:
            self.m1y = torch.squeeze(torch.matmul(self.H_PO, self.m1x_prior))
            # Predict the 2-nd moment of y
            self.m2y = torch.matmul(self.H_PO, self.m2x_prior)
            self.m2y = torch.matmul(self.m2y, self.H_PO.T) + self.R_PO

        if obs_model == 1:
            self.m1y = torch.squeeze(torch.matmul(self.H_AO, self.m1x_prior))
            # Predict the 2-nd moment of y
            self.m2y = torch.matmul(self.H_AO, self.m2x_prior)
            self.m2y = torch.matmul(self.m2y, self.H_AO.T) + self.R_AO

        if obs_model == 2:
            self.m1y = torch.squeeze(torch.matmul(self.H_PAO, self.m1x_prior))
            # Predict the 2-nd moment of y
            self.m2y = torch.matmul(self.H_PAO, self.m2x_prior)
            self.m2y = torch.matmul(self.m2y, self.H_PAO.T) + self.R_PAO

    # Compute the Kalman Gain
    def KGain(self, obs_model):

        if obs_model == 0:
            self.KG = torch.matmul(self.m2x_prior, self.H_PO.T)
        elif obs_model == 1:
            self.KG = torch.matmul(self.m2x_prior, self.H_AO.T)
        elif obs_model == 2:
            self.KG = torch.matmul(self.m2x_prior, self.H_PAO.T)

        self.KG = torch.matmul(self.KG, torch.inverse(self.m2y))

        # Save KalmanGain
        #self.KG_array[self.i] = self.KG
        self.i += 1

    # Innovation
    def Innovation(self, y):
        self.dy = y - self.m1y

    # Compute Posterior
    def Correct(self):
        # Compute the 1-st posterior moment
        self.m1x_posterior = self.m1x_prior + torch.matmul(self.KG, self.dy)

        # Compute the 2-nd posterior moment
        self.m2x_posterior = torch.matmul(self.m2y, torch.transpose(self.KG, 0, 1))
        self.m2x_posterior = self.m2x_prior - torch.matmul(self.KG, self.m2x_posterior)


    def InitSequence(self, m1x_0, m2x_0):
        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0


    #########################
    ### Generate Sequence ###
    #########################

    def GenerateSequence(self, input_pos, input_acc, T, time_list, time_index_list):
        # Pre allocate an array for predicted state and variance
        self.x = torch.empty(size=[self.m, T])
        self.sigma = torch.empty(size=[self.m, self.m, T])
        # Pre allocate KG array
        self.KG_array = torch.zeros((T, self.m, self.n))
        self.i = 0  # Index for KG_array alocation

        self.m1x_posterior = torch.squeeze(self.m1x_0)
        self.m2x_posterior = torch.squeeze(self.m2x_0)


        index = 0
        pos_index = 0
        acc_index = 0
        count_obs = 0
        for t in tqdm(range(0, T)):

            # check if we have observation
            if t == time_list[index]:
                count_obs += 1
                # Check what kind of observation
                if time_index_list[index] == 0:
                    # We have position observation
                    y_pos = input_pos[pos_index]  # Extract Position observation
                    pos_index += 1
                    index += 1
                    # Convert to tensor
                    y = torch.zeros(2)
                    y[0] = y_pos[0]
                    y[1] = y_pos[1]
                    # KF step
                    self.Predict(0)
                    self.KGain(0)
                    self.Innovation(y)
                    self.Correct()
                    self.x[:, t] = torch.squeeze(self.m1x_posterior)
                    self.sigma[:, :, t] = torch.squeeze(self.m2x_posterior)

                elif time_index_list[index] == 1:
                    # We have acceleration observation
                    y_acc = input_acc[acc_index]  # Extract acceleration observation
                    acc_index += 1
                    index += 1
                    # Convert to tensor
                    y = torch.zeros(2)
                    y[0] = y_acc[0]
                    y[1] = y_acc[1]
                    # KF step
                    self.Predict(1)
                    self.KGain(1)
                    self.Innovation(y)
                    self.Correct()
                    self.x[:, t] = torch.squeeze(self.m1x_posterior)
                    self.sigma[:, :, t] = torch.squeeze(self.m2x_posterior)

                elif time_index_list[index] == 2:
                    # We have acceleration & position observation
                    y_acc = input_acc[acc_index]  # Extract acceleration observation
                    y_pos = input_pos[pos_index]  # Extract position observation
                    acc_index += 1
                    pos_index += 1
                    index += 1
                    # Convert to tensor
                    y = torch.zeros(4)
                    y[0] = y_pos[0]
                    y[1] = y_acc[0]
                    y[2] = y_pos[1]
                    y[3] = y_acc[1]
                    # KF step
                    self.Predict(2)
                    self.KGain(2)
                    self.Innovation(y)
                    self.Correct()
                    self.x[:, t] = torch.squeeze(self.m1x_posterior)
                    self.sigma[:, :, t] = torch.squeeze(self.m2x_posterior)

            # We don't have any observation
            else:
                # Only Prediction Steps
                self.Predict(0)
                self.KGain(0)
                # Save the prediction
                self.x[:, t] = torch.squeeze(self.m1x_prior)
                self.sigma[:, :, t] = torch.squeeze(self.m2x_prior)
                # We say that posterior = prior since we have no update step
                self.m1x_posterior = self.m1x_prior
                self.m2x_posterior = self.m2x_prior

        print("Index reached: ", index)
        print("Size of Observation arr: ", len(time_list))
        print("How many time we observe: ", count_obs)
        print("How many time we observe IMU: ", acc_index)
        print("How many time we observe GPS: ", pos_index)