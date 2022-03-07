import torch.nn as nn
import torch
import time
from src.mm_CTRA_h_full import getJacobian_F, getJacobian_H
import numpy as np
import src.parameters as params
ratio = params.ratio


class ExtendedKalmanFilter:

    def __init__(self, SystemModel):
        self.f = SystemModel.f
        self.m = SystemModel.m

        # Has to be transformed because of EKF non-linearity
        self.Q = SystemModel.Q

        self.h = SystemModel.h
        self.n = SystemModel.n

        # Has to be transofrmed because of EKF non-linearity
        self.R = SystemModel.R

        self.T = SystemModel.T
        self.T_test = SystemModel.T_test

        # Pre allocate KG array
        self.KG_array = torch.zeros((self.T_test, self.m, self.n))
        self.error = 0
        self.use_identity = 0
        self.calculated_inverse = 0


    # Predict
    def Predict(self):
        # Predict the 1-st moment of x
        #self.m1x_prior = torch.squeeze(self.f(self.m1x_posterior))
        self.m1x_prior = torch.squeeze(torch.matmul(getJacobian_F(self.m1x_posterior), self.m1x_posterior))
        # Compute the Jacobians
        self.UpdateJacobians(getJacobian_F(self.m1x_posterior), getJacobian_H(self.m1x_prior))
        # Predict the 2-nd moment of x
        self.m2x_prior = torch.matmul(self.F, self.m2x_posterior)
        self.m2x_prior = torch.matmul(self.m2x_prior, self.F_T) + self.Q(self.m1x_prior)

        # Predict the 1-st moment of y
        #self.m1y = torch.squeeze(self.h(self.m1x_prior))
        self.m1y = torch.squeeze(torch.matmul(getJacobian_H(self.m1x_prior),self.m1x_prior))
        # Predict the 2-nd moment of y
        self.m2y = torch.matmul(self.H, self.m2x_prior)
        self.m2y = torch.matmul(self.m2y, self.H_T) + self.R
        print("R matrix: ", self.R)

    # Compute the Kalman Gain
    def KGain(self):

        self.KG = torch.matmul(self.m2x_prior, self.H_T)  # Just self.m2x_prior except last row
        diff_matrix = self.KG[:5] - self.m2y
        print(torch.max(diff_matrix))
        if torch.max(diff_matrix) < 0.01:
            self.use_identity += 1
            self.KG = self.H_T
        else:
            try:
                inv = torch.inverse(self.m2y)
                self.KG = torch.matmul(self.KG, inv)
                self.calculated_inverse += 1
            except:
                self.error += 1
                self.KG = self.H_T

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

    def InitSequence(self, m1x_0, m2x_0):
        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0

        #########################

    def UpdateJacobians(self, F, H):
        self.F = F
        self.F_T = torch.transpose(F, 0, 1)
        self.H = H
        self.H_T = torch.transpose(H, 0, 1)
        # print(self.H,self.F,'\n')

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
            # Only Prediction Steps
            self.Predict()
            self.KGain()
            # Save the prediction
            xt, sigmat = self.m1x_prior, self.m2x_prior

            # If we have an observation
            if T%ratio == 9:
                # We perform update step
                yt = torch.squeeze(y[:, int((t+1)/ratio - 1)])  # Every 10th step we take a new observation
                self.Update(yt)
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
        print("Replaced KG with identity: ", self.error)
        print("Used Identity: ", self.use_identity)
        print("Calculated Inverse: ", self.calculated_inverse)



def EKFTest(SysModel, test_input, test_target, allStates=True):
    N_T = test_target.size()[0]

    # LOSS
    loss_fn = nn.MSELoss(reduction='mean')

    # MSE [Linear]
    MSE_EKF_linear_arr = torch.empty(N_T)

    EKF = ExtendedKalmanFilter(SysModel)
    EKF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)

    KG_array = torch.zeros_like(EKF.KG_array)
    EKF_out = torch.empty([N_T, SysModel.m, SysModel.T_test])
    start = time.time()
    for j in range(0, N_T):
        # We only receive a observation every 10th timestep
        EKF.GenerateSequence(test_input[j, :, :], EKF.T_test)

        if (allStates):
            MSE_EKF_linear_arr[j] = loss_fn(EKF.x, test_target[j, :, :]).item()
        else:
            loc = torch.tensor([True, False, True, False])
            MSE_EKF_linear_arr[j] = loss_fn(EKF.x[loc, :], test_target[j, :, :]).item()
        KG_array = torch.add(EKF.KG_array, KG_array)
        EKF_out[j, :, :] = EKF.x
    end = time.time()
    t = end - start
    # Average KG_array over Test Examples
    KG_array /= N_T

    MSE_EKF_linear_avg = torch.mean(MSE_EKF_linear_arr)
    MSE_EKF_dB_avg = 10 * torch.log10(MSE_EKF_linear_avg)

    # Standard deviation
    MSE_EKF_dB_std = torch.std(MSE_EKF_linear_arr, unbiased=True)
    MSE_EKF_dB_std = 10 * torch.log10(MSE_EKF_dB_std)

    print("EKF - MSE LOSS:", MSE_EKF_dB_avg, "[dB]")
    print("EKF - MSE STD:", MSE_EKF_dB_std, "[dB]")
    # Print Run Time
    print("Inference Time:", t)

    return [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, KG_array, EKF_out]