import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from NN_parameters import weights
import time



def custom_loss(output, target):
    # Input: [mxT]
    # Tries to normalize each output based on its range
    loss_MSE = nn.MSELoss(reduction='mean')
    LOSS = loss_MSE(weights*output,weights*target)
    return LOSS

def NNTest(SysModel, test_input, test_target, init_conditions, path_results):


    if torch.cuda.is_available():
        cuda0 = torch.device("cuda:0") # you can continue going on here, like cuda:1 cuda:2....etc. 
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else :
        cuda0 = torch.device("cpu")

    # N_T  is the number of test samples
    N_T = test_input.size()[0]
    # T_test is the trajectory size
    T_test = test_input.size()[2]
    MSE_test_linear_arr = np.empty([N_T])
    MSE_test_linear_dim = torch.empty((N_T,SysModel.m))

    # MSE LOSS Function
    #loss_fn = nn.MSELoss(reduction='mean')
    loss_fn = custom_loss

    Model = torch.load(path_results+'best-model.pt')

    Model.eval()
    torch.no_grad()

    KGain_array = torch.zeros((T_test, Model.m, Model.n))
    x_out_array = torch.empty(N_T,SysModel.m, T_test)
    y_processed = torch.empty(N_T,SysModel.n, T_test)

    # We iterate over every test sample
    for j in range(0, N_T):
        Model.i = 0
        # Unrolling Forward Pass
        # Initialize next sequence
        v_0 = torch.unsqueeze(init_conditions[j,:],dim=0).T
        Model.InitSequence(v_0, SysModel.m2x_0, T_test)
        
        y_mdl_tst = test_input[j, :, :]
        y_processed[j,:,0:1], y_acc_test = Model.preprocess(SysModel.m1x_0,y_mdl_tst[:,0:1])
        forward_acc_test = y_acc_test


        x_Net_mdl_tst = torch.empty(SysModel.m, T_test)
        vel_test = torch.empty(2, T_test)
        frequencies = torch.empty(T_test)

        # We iterate over the time ( trajectory )
        for t in range(0, T_test):
            start_time = time.time()

            # We perform a forward pass in the Model to get the vel_posterior
            # TODO: MISSING ARGUMENT ACC ? WHY IS ACC NEEDED (Contains the target values (do we need it for prediciton step?)
            forward_y_test = torch.squeeze(y_processed[j,:,t:t+1])



            # TODO: The model only return v_x and v_y but later onward we try to extract a_x, a_y, yaw_rate?!
            # This save the velocity v_x and v_y
            vel_test[: ,t] = Model(forward_y_test, forward_acc_test)
            # Here we concatenate the acc and vel together to get the state vector
            x_Net_mdl_tst[:, t:t+1] = torch.cat((forward_acc_test.reshape(3,1),vel_test[:,t:t+1]),dim=0)
            #x_Net_mdl_tst[:,t] = Model(forward_y_test, forward_acc_test)



            # Preprocessing: Calibration + Frame transormation + Bias removal
            if(t+1 != T_test):
                y_processed[j,:,t+1:t+2], forward_acc_test = Model.preprocess(x_Net_mdl_tst[:,t:t+1],y_mdl_tst[:,t+1:t+2])

            # This is used to calculate the inference time and frequency
            frequencies[t] = 1/(time.time()-start_time)
            #print(frequencies[t])

        # We normalize our results
        norm_vel = func.normalize(vel_test, p=2, dim=1, eps=1e-12, out=None)
        norm_x_test = func.normalize(x_Net_mdl_tst, p=2, dim=0, eps=1e-12, out=None)
        test_target_norm = func.normalize(test_target[j,:,:], p=2, dim=0, eps=1e-12, out=None)

        # We calculate our Loss for every test sample
        MSE_test_linear_arr[j] = loss_fn(norm_x_test, test_target_norm).item()
        x_out_array[j,:,:] = x_Net_mdl_tst

        # Loss over separate dimensions
        # All of these correspond to loss over v_x, v_y and so on. Not well deocumented
        # TODO: Figure out what each dimension represents
        MSE_test_linear_dim[j,0] = loss_fn(x_Net_mdl_tst[0,:], test_target[j, 0, :]).item() # v_x this is all guessed
        MSE_test_linear_dim[j,1] = loss_fn(x_Net_mdl_tst[1,:], test_target[j, 1, :]).item() # v_y
        MSE_test_linear_dim[j,2] = loss_fn(x_Net_mdl_tst[2,:], test_target[j, 2, :]).item() # yaw_rate
        MSE_test_linear_dim[j,3] = loss_fn(x_Net_mdl_tst[3,:], test_target[j, 3, :]).item() # a_x
        MSE_test_linear_dim[j,4] = loss_fn(x_Net_mdl_tst[4,:], test_target[j, 4, :]).item() # a_y

        try:
            KGain_array = torch.add(Model.KGain_array, KGain_array)
            KGain_array /= N_T
        except:
            KGain_array = None

    # Average all the losses over all test samples and calculate dB
    MSE_test_linear_avg = np.mean(MSE_test_linear_arr)
    MSE_test_dB_avg = 10 * np.log10(MSE_test_linear_avg)

    MSE_test_linear_dim_avg = torch.mean(MSE_test_linear_dim,dim=0)
    MSE_test_dB_dim_avg = 10 * torch.log10(MSE_test_linear_dim_avg)

    print("MSE Test : ",MSE_test_dB_avg," [dB]")

    return [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, MSE_test_linear_dim_avg, MSE_test_dB_dim_avg, KGain_array, x_out_array, y_processed,frequencies]