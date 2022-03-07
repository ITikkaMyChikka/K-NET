
import torch
import numpy as np
import src.parameters as params
from src.Linear_models.CTRA_mm import F_jacobian_smooth

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    dev = torch.device("cpu")

class SystemModel:

    def __init__(self, F, Q, H, R, T, prior_Q=None, prior_Sigma=None, prior_S=None):

        ####################
        ### Motion Model ###
        ####################
        self.F = F

        self.Q = Q
        self.m = self.Q.size()[0]

        #########################
        ### Observation Model ###
        #########################
        self.H = H

        self.R = R
        self.n = self.R.size()[0]

        ################
        ### Sequence ###
        ################
        # Assign T
        self.T = T

        #########################
        ### Covariance Priors ###
        #########################
        if prior_Q is None:
            self.prior_Q = torch.eye(self.m)
        else:
            self.prior_Q = prior_Q

        if prior_Sigma is None:
            self.prior_Sigma = torch.zeros((self.m, self.m))
        else:
            self.prior_Sigma = prior_Sigma

        if prior_S is None:
            self.prior_S = torch.eye(self.n)
        else:
            self.prior_S = prior_S



    #####################
    ### Init Sequence ###
    #####################
    def InitSequence(self, m1x_0, m2x_0):

        self.m1x_0 = m1x_0
        self.x_prev = m1x_0
        self.m2x_0 = m2x_0


    #########################
    ### Update Covariance ###
    #########################
    def UpdateCovariance_Gain(self, q, r):

        self.q = q
        self.Q = q * q * torch.eye(self.m)

        self.r = r
        self.R = r * r * torch.eye(self.n)

    def UpdateCovariance_Matrix(self, Q, R):

        self.Q = Q

        self.R = R


    #########################
    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, Q_gen, R_gen, T):
        # Pre allocate an array for current state
        self.x = torch.empty(size=[self.m, T])
        # Pre allocate an array for current observation
        self.y = torch.empty(size=[self.n, T])
        # Set x0 to be x previous
        self.x_prev = self.m1x_0

        xt = self.x_prev

        # Generate Sequence Iteratively
        for t in range(0, T):

            ########################
            #### State Evolution ###
            ########################
            xt = torch.matmul(self.F, self.x_prev)
            #xt = F_jacobian_smooth(self.x_prev)

            # Add noise to the acceleration with variance = q
            mean = torch.zeros(1)
            w_x = np.random.normal(mean, params.q_ca**0.5, 1)
            w_y = np.random.normal(mean, params.q_ca**0.5, 1)

            # Additive Process Noise
            xt[2] = xt[2]+w_x  # add noise to the acceleration
            xt[5] = xt[5]+w_y

            ################
            ### Emission ###
            ################
            yt = torch.matmul(self.H, xt)

            """
            # Observation Noise
            mean = torch.zeros(self.n)
            er = np.random.multivariate_normal(mean, R_gen, 1)
            er = torch.transpose(torch.tensor(er), 0, 1)
            er = er.reshape(self.n)

            # Additive Observation Noise
            yt = yt.add(er)
            """
            ########################
            ### Squeeze to Array ###
            ########################

            # Save Current State to Trajectory Array
            self.x[:, t] = torch.squeeze(xt)

            # Save Current Observation to Trajectory Array
            self.y[:, t] = torch.squeeze(yt)

            ################################
            ### Save Current to Previous ###
            ################################
            self.x_prev = xt


    ######################
    ### Generate Batch ###
    ######################
    def GenerateBatch(self, size, T):

        # Allocate Empty Array for Input
        self.Input = torch.empty(size, self.n, T)

        # Allocate Empty Array for Target
        self.Target = torch.empty(size, self.m, T)

        ### Generate Examples

        for i in range(0, size):
            # Generate Sequence
            print("generating sample: ", i)
            self.InitSequence(self.m1x_0, self.m2x_0)
            self.GenerateSequence(self.Q, self.R, T)

            # Training sequence input
            self.Input[i, :, :] = self.y

            # Training sequence output
            self.Target[i, :, :] = self.x





class SystemModel_KITTI:

    def __init__(self, F, Q, H_PO, R_PO, H_AO, R_AO, H_PAO, R_PAO, T, prior_Q=None, prior_Sigma=None, prior_S=None):

        ####################
        ### Motion Model ###
        ####################
        self.F = F

        self.Q = Q
        self.m = self.Q.size()[0]

        #########################
        ### Observation Model ###
        #########################


        self.H_PO = H_PO
        self.R_PO = R_PO
        self.H_AO = H_AO
        self.R_AO = R_AO
        self.H_PAO = H_PAO
        self.R_PAO = R_PAO

        self.n = self.R_PO.size()[0]

        ################
        ### Sequence ###
        ################
        # Assign T
        self.T = T

        #########################
        ### Covariance Priors ###
        #########################
        if prior_Q is None:
            self.prior_Q = torch.eye(self.m)
        else:
            self.prior_Q = prior_Q

        if prior_Sigma is None:
            self.prior_Sigma = torch.zeros((self.m, self.m))
        else:
            self.prior_Sigma = prior_Sigma

        if prior_S is None:
            self.prior_S = torch.eye(self.n)
        else:
            self.prior_S = prior_S



    #####################
    ### Init Sequence ###
    #####################
    def InitSequence(self, m1x_0, m2x_0):

        self.m1x_0 = m1x_0
        self.x_prev = m1x_0
        self.m2x_0 = m2x_0


    #########################
    ### Update Covariance ###
    #########################
    def UpdateCovariance_Gain(self, q, r):

        self.q = q
        self.Q = q * q * torch.eye(self.m)

        self.r = r
        self.R = r * r * torch.eye(self.n)

    def UpdateCovariance_Matrix(self, Q, R):

        self.Q = Q

        self.R = R


    #########################
    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, Q_gen, R_gen, T):
        # Pre allocate an array for current state
        self.x = torch.empty(size=[self.m, T])
        # Pre allocate an array for current observation
        self.y = torch.empty(size=[self.n, T])
        # Set x0 to be x previous
        self.x_prev = self.m1x_0

        xt = self.x_prev

        # Generate Sequence Iteratively
        for t in range(0, T):

            ########################
            #### State Evolution ###
            ########################
            xt = torch.matmul(self.F, self.x_prev)

            # Add noise to the acceleration with variance = q
            mean = torch.zeros(1)
            w_x = np.random.normal(mean, params.q_ca**0.5, 1)
            w_y = np.random.normal(mean, params.q_ca**0.5, 1)

            # Additive Process Noise
            xt[2] = xt[2]+w_x  # add noise to the acceleration
            xt[5] = xt[5]+w_y

            ################
            ### Emission ###
            ################
            yt = torch.matmul(self.H, xt)

            """
            # Observation Noise
            mean = torch.zeros(self.n)
            er = np.random.multivariate_normal(mean, R_gen, 1)
            er = torch.transpose(torch.tensor(er), 0, 1)
            er = er.reshape(self.n)

            # Additive Observation Noise
            yt = yt.add(er)
            """
            ########################
            ### Squeeze to Array ###
            ########################

            # Save Current State to Trajectory Array
            self.x[:, t] = torch.squeeze(xt)

            # Save Current Observation to Trajectory Array
            self.y[:, t] = torch.squeeze(yt)

            ################################
            ### Save Current to Previous ###
            ################################
            self.x_prev = xt


    ######################
    ### Generate Batch ###
    ######################
    def GenerateBatch(self, size, T):

        # Allocate Empty Array for Input
        self.Input = torch.empty(size, self.n, T)

        # Allocate Empty Array for Target
        self.Target = torch.empty(size, self.m, T)

        ### Generate Examples

        for i in range(0, size):
            # Generate Sequence
            print("generating sample: ", i)
            self.InitSequence(self.m1x_0, self.m2x_0)
            self.GenerateSequence(self.Q, self.R, T)

            # Training sequence input
            self.Input[i, :, :] = self.y

            # Training sequence output
            self.Target[i, :, :] = self.x




















class SystemModel_NL:

    def __init__(self, F, Q, H, R, T, F_Jacobian=None, prior_Q=None, prior_Sigma=None, prior_S=None):

        ####################
        ### Motion Model ###
        ####################
        self.F = F
        self.F_Jacobian = F_Jacobian

        self.Q = Q
        self.m = 6

        #########################
        ### Observation Model ###
        #########################
        self.H = H

        self.R = R
        self.n = 6

        ################
        ### Sequence ###
        ################
        # Assign T
        self.T = T


        #########################
        ### Covariance Priors ###
        #########################
        if prior_Q is None:
            self.prior_Q = torch.eye(self.m)
        else:
            self.prior_Q = prior_Q

        if prior_Sigma is None:
            self.prior_Sigma = torch.zeros((self.m, self.m))
        else:
            self.prior_Sigma = prior_Sigma

        if prior_S is None:
            self.prior_S = torch.eye(self.n)
        else:
            self.prior_S = prior_S



    #####################
    ### Init Sequence ###
    #####################
    def InitSequence(self, m1x_0, m2x_0):

        self.m1x_0 = m1x_0
        self.x_prev = m1x_0
        self.m2x_0 = m2x_0


    #########################
    ### Update Covariance ###
    #########################
    def UpdateCovariance_Gain(self, q, r):

        self.q = q
        self.Q = q * q * torch.eye(self.m)

        self.r = r
        self.R = r * r * torch.eye(self.n)

    def UpdateCovariance_Matrix(self, Q, R):

        self.Q = Q

        self.R = R


    #########################
    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, Q_gen, R_gen, T):
        # Pre allocate an array for current state
        self.x = torch.empty(size=[self.m, T])
        # Pre allocate an array for current observation
        self.y = torch.empty(size=[self.n, T])
        # Set x0 to be x previous
        self.x_prev = self.m1x_0

        xt = self.x_prev

        # Generate Sequence Iteratively
        for t in range(0, T):

            ########################
            #### State Evolution ###
            ########################
            #xt = torch.matmul(F_jacobian_smooth(self.x_prev), self.x_prev)
            xt = self.F(self.x_prev)
            # xt = torch.matmul(self.F, self.x_prev)

            # Add noise to the acceleration with variance = q
            #R = torch.tensor([[params.sa_gen, 0], [0, params.sw_gen]])
            R = torch.tensor([[0.5**2, 0], [0, (1.0*params.T_ctra)**2]])
            mean = torch.zeros(2)
            er = np.random.multivariate_normal(mean, R, 1)
            er = torch.transpose(torch.tensor(er), 0, 1)
            er = er.reshape(2)

            #print(er)

            # Additive Process Noise
            xt[4] = xt[4]+er[0]  # add noise to the acceleration
            xt[5] = xt[5]+er[1]  # add noise to yaw_rate

            ################
            ### Emission ###
            ################
            yt = torch.matmul(self.H(xt), xt)

            """
            # Observation Noise
            mean = torch.zeros(self.n)
            er = np.random.multivariate_normal(mean, R_gen, 1)
            er = torch.transpose(torch.tensor(er), 0, 1)
            er = er.reshape(self.n)

            # Additive Observation Noise
            yt = yt.add(er)
            """
            ########################
            ### Squeeze to Array ###
            ########################

            # Save Current State to Trajectory Array
            self.x[:, t] = torch.squeeze(xt)

            # Save Current Observation to Trajectory Array
            self.y[:, t] = torch.squeeze(yt)

            ################################
            ### Save Current to Previous ###
            ################################
            self.x_prev = xt


    ######################
    ### Generate Batch ###
    ######################
    def GenerateBatch(self, size, T):

        # Allocate Empty Array for Input
        self.Input = torch.empty(size, self.n, T)

        # Allocate Empty Array for Target
        self.Target = torch.empty(size, self.m, T)

        ### Generate Examples

        for i in range(0, size):
            # Generate Sequence
            print("generating sample: ", i)
            self.InitSequence(self.m1x_0, self.m2x_0)
            self.GenerateSequence(self.Q, self.R, T)

            # Training sequence input
            self.Input[i, :, :] = self.y

            # Training sequence output
            self.Target[i, :, :] = self.x
