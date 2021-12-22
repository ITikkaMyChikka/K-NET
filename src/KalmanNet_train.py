import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
import random
from NN_parameters import weights
from KalmanNet_plt import plotTrajectories



def custom_loss(output, target):
    # Input: [mxT]
    # Tries to normalize each output based on its range
    loss_MSE = nn.MSELoss(reduction='mean')
    LOSS = loss_MSE(output,target)
    return LOSS

def NNTrain(SysModel, Model, cv_input, cv_target, train_input, train_target, init_conditions_train, init_conditions_cv, 
            N_Epochs, N_B, learning_rate, wd, path_results,dynamic_training_num=0):

    if(dynamic_training_num==0):
        N_E = train_input.size()[0]
        N_CV = cv_input.size()[0]
        print('N_E:',N_E,' N_CV:',N_CV,' N_B:',N_B)
        MSE_cv_linear_batch = np.empty([N_CV])

    # MSE LOSS Function
    #loss_fn = nn.MSELoss(reduction='mean')
    loss_fn = custom_loss

    # Use the optim package to define an Optimizer that will update the weights of
    # the model for- us. Here we will use Adam; the optim package contains many other
    # optimization algorithms. The first argument to the Adam constructor tells the
    # optimizer which Tensors it should update.
    optimizer = torch.optim.Adam(Model.parameters(), lr=learning_rate, weight_decay=wd)

    MSE_cv_linear_epoch = np.empty([N_Epochs])
    MSE_cv_dB_epoch = np.empty([N_Epochs])

    MSE_train_linear_batch = np.empty([N_B])
    MSE_train_linear_epoch = np.empty([N_Epochs])
    MSE_train_dB_epoch = np.empty([N_Epochs])


    ##############
    ### Epochs ###
    ##############

    MSE_cv_dB_opt = 1000
    MSE_cv_idx_opt = 0
    n_e = 0

    # ti tells us in which epoch we are
    for ti in range(0, N_Epochs):

        #################################
        ### Validation Sequence Batch ###
        #################################

        if(dynamic_training_num != 0):
            # We will do dynamic training
            if(ti == 0):
                # In the first epoch we use the input as it is
                # Only reason why this is so ugly is so I don't have to modify the entire code for dynamic training
                ind = 0
                cv_input_array = cv_input
                train_input_array = train_input

                cv_target_array = cv_target
                train_target_array = train_target

                init_conditions_train_array = init_conditions_train
                init_conditions_cv_array = init_conditions_cv
            if(ti == round(N_Epochs/dynamic_training_num)*ind):
                # The larger the epoch is the longer the training sequence length will be through ind
                print("\nIncreasing training sequence length...\n")
                cv_input = cv_input_array[ind]
                train_input = train_input_array[ind]
                cv_target = cv_target_array[ind]
                train_target = train_target_array[ind]
                init_conditions_cv = init_conditions_cv_array[ind]
                init_conditions_train = init_conditions_train_array[ind]

                N_E = train_input.size()[0]
                N_CV = cv_input.size()[0]
                print('N_E:',N_E,' N_CV:',N_CV,' N_B:',N_B)
                MSE_cv_linear_batch = np.empty([N_CV])

                ind += 1


        T_sequence = cv_input.size(2)

        # Cross Validation Mode
        Model.eval()
        y_processed_cv = torch.zeros(N_CV,SysModel.n, T_sequence)
        cv_plot_ind = 5
        train_plot_ind = random.randint(0, N_B - 1)

        if(ti%5==0):
            a=0

        # In every epoch we iterate through the N_CV
        # N_CV = Number of Cross Validation Examples
        for j in range(0, N_CV):
            Model.i = 0 # Don't know what this is
            
            # Initialize next sequence
            v_0 = torch.unsqueeze(init_conditions_cv[j,:],dim=0).T # init_conditions_cv is an input, we basically take slices of it and set it to v_0
            Model.InitSequence(v_0, SysModel.m2x_0, T_sequence)
            # v_0 is being set as m1x_posterior, state_process_posterior_0 and as m1x_prior_previous
            # The T sequence is important for size I am guessing

            if (j==cv_plot_ind):
                #print('cv_input_5:\n',cv_input[j, :, :5])
                #print('y_processed_cv_5:\n',y_processed_cv[j,:,:5])
                pass

            # Preprocessing: Calibration + Frame transormation + Bias removal
            y_processed_cv[j,:,0:1], y_acc_cv = Model.preprocess(v_0,cv_input[j,:,0:1])
            forward_acc = y_acc_cv  # I am using the average of both IMU sensors to estimate the a_x, a_y and yaw_rate ( thorugh preprocess function)

            x_Net_cv = torch.empty(SysModel.m, T_sequence)
            vel_cv = torch.empty(2, T_sequence)
            # print('\n\nv_0:',v_0.T)
            
            for t in range(0, T_sequence):
                # We iterate over the time sequences, so every t we estimate the velocity
                forward_y = torch.squeeze(y_processed_cv[j,:,t:t+1])
                #forward_acc = torch.squeeze(cv_target[j,0:3,t:t+1]) # Why are we using here cv_target and not y_acc from Model.preprocess

                # We perfom one forward pass in the Model, which returns us vel_posterior
                vel_cv[:,t] = Model(forward_y, forward_acc)
                # We concatenate the cv target and our vel_posterior into one variable called x_Net_cv
                # This corresponds to the update step
                #x_Net_cv[:,t:t+1] = torch.cat((cv_target[j,0:3,t:t+1],vel_cv[:,t:t+1]),dim=0)
                x_Net_cv[:, t:t + 1] = torch.cat((forward_acc.reshape(3,1), vel_cv[:, t:t + 1]), dim=0)
                # Preprocessing: Calibration + Frame transormation + Bias removal
                #print('1:',cv_target[j,:,t].T,'\n2:',y_processed.T, '\n3:',x_Net_cv[:,t].T)


                # As long as we havent reached the end we calculate the new y which requires the cv_input and our previous results
                # this corresponds to the prediction step
                if(t+1 != T_sequence):
                    y_processed_cv[j,:,t+1:t+2], forward_acc = Model.preprocess(x_Net_cv[:,t:t+1],cv_input[j,:,t+1:t+2])
            
            # Clamp network output
            #x_Net_cv[torch.isnan(x_Net_cv)] = 0
            #x_Net_cv = torch.clamp(x_Net_cv, min=-100, max=100)

            # COMPUTE VALIDATION LOSS

            # We normalize our results
            norm_x_cv = func.normalize(x_Net_cv, p=2, dim=1, eps=1e-12, out=None)
            norm_vel = func.normalize(vel_cv, p=2, dim=1, eps=1e-12, out=None)
            cv_target_norm = func.normalize(cv_target[j,:,:], p=2, dim=1, eps=1e-12, out=None)

            # We calculate our loss with our custom loss function ( which is actually MSE )
            # for every batch, where 1 batch contains the entire Number of Cross Validation Examples
            MSE_cv_linear_batch[j] = loss_fn(norm_vel, cv_target_norm[-2:,:]).item()

            # cv_plot_ind = 5 so in every epoch we iterate through j {cv_input size} and when it reaches 5 we plot the trajectories
            if (j==cv_plot_ind):
                #print('cv_input_5:\n',cv_input[j, :, :5])
                #print('y_processed_cv_5:\n',y_processed_cv[j,:,:5])
                plotTrajectories(x_Net_cv,cv_target[j,:,:],time=None,sensors=y_processed_cv[j,:,:],position=None,file_name='trajectories_cv.png',dpi=60)
                #print('x_cv:\n',x_Net_cv,'\ninit_cv:\n',v_0,'\ny_processed:\n',y_processed_cv[j,:,:],'\nMKF:\n',cv_target[j,:,:])

        # After iterating through cv_input size
        # Average our loss and save it as dB for each epoch
        MSE_cv_linear_epoch[ti] = np.mean(MSE_cv_linear_batch)
        MSE_cv_dB_epoch[ti] = 10 * np.log10(MSE_cv_linear_epoch[ti])

        ###############################
        ### Training Sequence Batch ###
        ###############################

        # Training Mode
        Model.train()

        # Init Hidden State
        # Sets the weights to zero
        Model.init_hidden()

        Batch_Optimizing_LOSS_sum = 0
        y_processed_train = torch.zeros(N_B,SysModel.n, T_sequence)

        # N_B stands for Number of Batches
        for j in range(0, N_B):
            Model.i = 0 # Don't know this
            # N_E = Number of Training Samples
            n_e = random.randint(0, N_E - 1)
            #n_e = (n_e + 1)%N_E 

            # Initiliazation: We pick a random training sample as initial value
            v_0 = torch.unsqueeze(init_conditions_train[n_e,:],dim=0).T
            Model.InitSequence(v_0, SysModel.m2x_0, T_sequence)

            # Preprocessing: Calibration + Frame transormation + Bias removal
            y_processed_train[j,:,0:1], y_acc_train = Model.preprocess(v_0, train_input[n_e, :, 0:1])
            forward_acc_train = y_acc_train # We use the average of both IMU sensor to estimate the acceleration and yaw

            x_Net_training = torch.empty(SysModel.m, T_sequence)
            vel_train = torch.empty(2, T_sequence)

            # We start the training by iterating through the time sequence
            for t in range(0, T_sequence):

                forward_y_train = torch.squeeze(y_processed_train[j,:,t:t+1])
                #forward_acc_train = torch.squeeze(train_target[n_e,0:3,t:t+1])

                # In every time step t, we perfom a forward pass with the network which return vel_posterior
                # This is the update Step
                vel_train[:,t] = Model(forward_y_train, forward_acc_train)

                # We save the results in x_Net_training which contains the train_target and vel_posterior results
                x_Net_training[:,t:t+1] = torch.cat((train_target[n_e,0:3,t:t+1],vel_train[:,t:t+1]),dim=0)
                #print('1:',train_target[n_e,:,t].T,'\n2:',y_processed.T, '\n3:',x_Net_training[:,t].T)
                # Preprocessing: Calibration + Frame transormation + Bias removal
                if(t+1 != T_sequence):
                    # This corresponds to the prediction step
                    y_processed_train[j,:,t+1:t+2], forward_acc_train = Model.preprocess(x_Net_training[:,t:t+1], train_input[n_e,:,t+1:t+2])

            # Clamp output
            #x_Net_training[torch.isnan(x_Net_training)] = 0
            #x_Net_training = torch.clamp(x_Net_training, min=-100, max=100)

            # Compute Training Loss

            # We normalize the values
            norm_x_train = func.normalize(x_Net_training, p=2, dim=1, eps=1e-12, out=None)
            norm_vel = func.normalize(vel_train, p=2, dim=1, eps=1e-12, out=None)
            train_target_norm_vel = func.normalize(train_target[n_e,-2:,:], p=2, dim=1, eps=1e-12, out=None)

            # We calculate the loss which is MSE
            LOSS = loss_fn(norm_vel, train_target_norm_vel)

            # We save the loss for every batch
            MSE_train_linear_batch[j] = LOSS.item()

            Batch_Optimizing_LOSS_sum = Batch_Optimizing_LOSS_sum + LOSS

            # After 5 batches (150 training samples) we plot the Trajectories
            if (j==train_plot_ind):
                plotTrajectories(x_Net_training,train_target[n_e,:,:],time=None,sensors=y_processed_train[j,:,:],position=None,file_name='trajectories_train.png',dpi=60)
                #print('\n\nx_train:\n',x_Net_training,'\ninit_cv:\n',v_0,'\ny_processed:\n',y_processed_train[j,:,:],'\nMKF:\n',train_target[n_e,:,:])


        # Average our loss over all batches and save the result for each epoch and calculate the dB
        MSE_train_linear_epoch[ti] = np.mean(MSE_train_linear_batch)
        MSE_train_dB_epoch[ti] = 10 * np.log10(MSE_train_linear_epoch[ti])

        ##################
        ### Optimizing ###
        ##################

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        Batch_Optimizing_LOSS_mean = Batch_Optimizing_LOSS_sum / N_B
        Batch_Optimizing_LOSS_mean.backward()

        # Gradient clipping to avoid exploding gradients
        nn.utils.clip_grad_norm_(Model.parameters(), max_norm=2.0, norm_type=2)
        #nn.utils.clip_grad_value_(Model.parameters(), clip_value=10)

        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()

        # Save best model
        # MSE_cv_dB_opt = 1000
        if(MSE_cv_dB_epoch[ti] < MSE_cv_dB_opt):

            MSE_cv_dB_opt = MSE_cv_dB_epoch[ti]
            MSE_cv_idx_opt = ti
            # TODO: FileNotFoundError: [Errno 2] No such file or directory: 'Results/Simulation_1/Results/best-model.pt'
            # I will save the Model in a new Folder
            #results_path = "/Users/sidhu/Documents/ETH/Semester Project/Adria_KN/src/Sim1/"
            #torch.save(Model, results_path + 'best-model.pt')
            torch.save(Model, path_results+'best-model.pt')

        ########################
        ### Training Summary ###
        ########################

        # We print the MSE for training and cross-validation for every epoch
        print(ti, "MSE Training :", MSE_train_dB_epoch[ti], "[dB]", "MSE Validation :", MSE_cv_dB_epoch[ti], "[dB]")

        if (ti > 1):
            d_train = MSE_train_dB_epoch[ti] - MSE_train_dB_epoch[ti - 1]
            d_cv    = MSE_cv_dB_epoch[ti] - MSE_cv_dB_epoch[ti - 1]
            print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")

        print("Optimal idx:", MSE_cv_idx_opt, "Optimal :", MSE_cv_dB_opt, "[dB]")

    return [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch]