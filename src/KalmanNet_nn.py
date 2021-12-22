"""# **Class: KalmanNet**"""

import torch
import torch.nn as nn
import torch.nn.functional as func

class KalmanNetNN(torch.nn.Module):

    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        super().__init__()

    ######################################
    ### Initialize Kalman Gain Network ###
    ######################################
    def InitKGainNet(self, H1, H2, nGRU):

        # Input Dimensions (+1 for time input)
        #D_in = self.m + self.n*2  #+ 1# x(t-1), y(t)
        D_in = self.m + self.n

        # Output Dimensions
        D_out = self.m * self.n;  # Kalman Gain

        ###################
        ### Input Layer ###
        ###################
        # Linear Layer
        self.KG_l1 = torch.nn.Linear(D_in, H1, bias=True)

        # ReLU (Rectified Linear Unit) Activation Function
        self.KG_relu1 = torch.nn.ReLU()

        ###########
        ### GRU ###
        ###########
        # Input Dimension
        self.input_dim = H1
        # Hidden Dimension
        self.hidden_dim = ((self.n * self.n) + (self.m * self.m)) #* 10 * 1
        # Number of Layers
        self.n_layers = nGRU
        # Batch Size
        self.batch_size = 1
        # Input Sequence Length
        self.seq_len_input = 1
        # Hidden Sequence Length
        self.seq_len_hidden = self.n_layers

        # batch_first = False
        # dropout = 0.1 ;

        # Initialize a Tensor for GRU Input
        # self.GRU_in = torch.empty(self.seq_len_input, self.batch_size, self.input_dim)

        # Initialize a Tensor for Hidden State
        self.hn = torch.randn(self.seq_len_hidden, self.batch_size, self.hidden_dim)

        # Initialize GRU Layer
        self.rnn_GRU = nn.GRU(input_size=self.input_dim,
                              hidden_size=self.hidden_dim, # The number of features in the hidden state h
                              num_layers=self.n_layers) # Number of recurrent layers

        ####################
        ### Hidden Layer ###
        ####################
        self.KG_l2 = torch.nn.Linear(self.hidden_dim, H2, bias=True)

        # ReLU (Rectified Linear Unit) Activation Function
        self.KG_relu2 = torch.nn.ReLU()

        ####################
        ### Output Layer ###
        ####################
        self.KG_l3 = torch.nn.Linear(H2, D_out, bias=True)

    ##################################
    ### Initialize System Dynamics ###
    ##################################
    def InitSystemDynamics(self, f, h, m, n):
        
        # Set State Evolution Function
        # X_t = X_t_1 * f + m
        self.f = f # evolution matrix F
        self.m = m # unknown covariance matrix (noise?)

        # Set Observation Function
        # Y_t =  h(X_t) + n
        self.h = h # Observation function
        self.n = n # unknown covariance matrix R (noise)

    ###########################
    ### Initialize Sequence ###
    ###########################
    def InitSequence(self, M1_0, M2_0, T):

        # Posteriior covariance matrix for the Evolution Satte Space
        # Actually v_0 I think since this is for the Initialization
        self.m1x_posterior = M1_0

        self.T = T # Size of Trajectory
        self.x_out = torch.empty(self.m, T)

        self.state_process_posterior_0 = M1_0
        self.m1x_posterior = M1_0
        self.m1x_prior_previous = self.m1x_posterior

        # KGain saving
        self.KGain_array = self.KG_array = torch.zeros((0,self.m,self.n))
        self.y_prev = torch.zeros(self.n)

    ######################
    ### Compute Priors ###
    ######################
    def step_prior(self):
        # Predict the 1-st moment of x
        # We feed the posterior result of step t-1 to our state evolution function f to obtain the prior results at step t
        # this corresponds to the prediction Step of the KF
        self.m1x_prior = self.f(self.m1x_posterior)

        # Predict the 1-st moment of y
        self.m1y = self.h(self.m1x_prior)

        self.state_process_prior_0 = self.f(self.state_process_posterior_0)
        self.obs_process_0 = self.h(self.state_process_prior_0)

    ##############################
    ### Kalman Gain Estimation ###
    ##############################
    def step_KGain_est(self, y):
        # Reshape and Normalize m1x Posterior
        # self.m1x_posterior[-2:,:] are the last 2 entries which are v_x and v_y
        with torch.no_grad():
            m1x_post_0 = self.m1x_posterior[-2:,:] - self.m1x_prior_previous[-2:,:] # Option 2
        #m1x_post_0 = self.m1x_prior
        #m1x_reshape = torch.squeeze(self.m1x_posterior) # Option 3
        m1x_reshape = torch.squeeze(m1x_post_0)
        #m1x_reshape = m1x_post_0
        m1x_norm = func.normalize(m1x_reshape, p=2., dim=0, eps=1e-12, out=None)

        # Normalize y
        #my_0 = y - torch.squeeze(self.obs_process_0) # Option 1
        with torch.no_grad():
            my_0 = y[-2:] - torch.squeeze(self.m1y[-2:,:]) # Option 2
        #my_0 = torch.squeeze(y)
        y_norm = func.normalize(my_0, p=2., dim=0, eps=1e-12, out=None)
        #y_norm = func.normalize(y, p=2, dim=0, eps=1e-12, out=None);

        # Additional input
        with torch.no_grad():
            my_1 = y[-2:] - self.y_prev[-2:]
        my1_norm = func.normalize(my_1, p=2., dim=0, eps=1e-12, out=None)
        self.y_prev = y

        # Input for counting
        #count_norm = func.normalize(torch.tensor([self.i]).float(),dim=0, eps=1e-12,out=None)

        # KGain Net Input
        #KGainNet_in = torch.cat([m1x_norm, y_norm, my1_norm], dim=0)
        KGainNet_in = torch.cat([m1x_norm, y_norm], dim=0)

        # Kalman Gain Network Step
        KG = self.KGain_step(KGainNet_in)

        # Reshape Kalman Gain to a Matrix
        self.KGain = torch.reshape(KG, (self.m, self.n))
        #### Debugging ####
        #self.KGain[3:4,:] = 0

    #######################
    ### Kalman Net Step ###
    #######################
    """
    INPUT:
    y: Sensor data
    acc = GT of velocity estimates
    
    OUTPUT:
    velocity posterior
    """
    def KNet_step(self, y,acc):

        # PREDICTION STEP
        # Compute Priors
        self.step_prior()

        # PREPERATION FOR UPDATE STEP
        # We need Kalman gain and innovation process to perform update step
        # Compute Kalman Gain
        self.step_KGain_est(y)

        # Save KGain in array
        self.KGain_array = torch.cat((self.KGain_array,torch.unsqueeze(self.KGain,dim=0)),dim=0)

        # Innovation
        y_obs = torch.unsqueeze(y, 1)

        with torch.no_grad():
            vel_prior = self.m1x_prior[-2:,:]
            dy = y_obs[-2:,:] - self.m1y[-2:,:] # Delta y is called innovation process

        # Now we are ready to perform the update step in the KF
        # We have the innovation process delta y and we have the K-Gain estimate and all the priors

        # Compute the 1-st posterior moment
        INOV = torch.matmul(self.KGain, dy) # Adding this to the velocity prior will result in vel_posterior

        # TODO: This is an in-place operation we have to make it not inplace
        # I fixed it by setting it to torch.no_grad() not sure if this is okay though
        # UPDATE STEP
        with torch.no_grad():
            self.m1x_posterior[0:3,0] = acc # this is the acceleration where we take values from cv_target, but why?
            vel_posterior = vel_prior + INOV*0.1 # Why 0.001? is this like a learning rate?
            self.m1x_posterior[-2:,:] = vel_posterior

        self.state_process_posterior_0 = self.state_process_prior_0
        self.m1x_prior_previous = self.m1x_prior

        # return
        return torch.squeeze(vel_posterior)

    ########################
    ### Kalman Gain Step ###
    ########################
    """
    INPUT
    KGainNet_in: Consists of the difference of 
    the observation and our predicted observation 
    concatenated with 
    the difference of the vel_prior and vel_posterior
    """
    def KGain_step(self, KGainNet_in):

        ###################
        ### Input Layer ###
        ###################
        # We feed it through a linear and then ReLu layer
        L1_out = self.KG_l1(KGainNet_in) # First linear layer,
        La1_out = self.KG_relu1(L1_out) # First ReLu layer

        ###########
        ### GRU ###
        ###########
        # Now we feed it into a GRU cell
        GRU_in = torch.empty(self.seq_len_input, self.batch_size, self.input_dim) # These just define the size
        GRU_in[0, 0, :] = La1_out
        #GRU_out, self.hn = self.rnn_GRU(GRU_in, self.hn) #Have to do it like this for transportation to c++
        rnn_GRU_result = self.rnn_GRU(GRU_in, self.hn)
        GRU_out = rnn_GRU_result[0]
        self.hn = rnn_GRU_result[1]
        GRU_out_reshape = torch.reshape(GRU_out, (1, self.hidden_dim))

        ####################
        ### Hidden Layer ###
        ####################
        # We feed the result of out GRU cell into another linear and ReLu layer
        L2_out = self.KG_l2(GRU_out_reshape)
        La2_out = self.KG_relu2(L2_out)

        ####################
        ### Output Layer ###
        ####################
        # We feed through a final linear layer and return the result
        # The output dimension of this layer corresponds to the K_gain Matrix
        L3_out = self.KG_l3(La2_out)
        return L3_out

    ###############
    ### Forward ###
    ###############
    def forward(self, y,acc):

        '''
        for t in range(0, self.T):
            self.x_out[:, t] = self.KNet_step(y[:, t])
        '''
        self.x_out = self.KNet_step(y,acc)

        return self.x_out

    #########################
    ### Init Hidden State ###
    #########################
    def init_hidden(self):
        weight = next(self.parameters()).data
        #hidden = weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_()
        hidden = weight.new_zeros(self.n_layers, self.batch_size, self.hidden_dim)
        self.hn = hidden.data