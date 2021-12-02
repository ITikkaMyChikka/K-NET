import torch
from datetime import datetime

def generate_params_file(N_E, N_CV, N_B, N_T, N_Epochs, lr, wd, T_train, T_test, nGRU, weights, dataset_file, results_path):
    
    file = open(results_path+"params.txt","w")
    # Title
    file.write("SIMULATION PARAMETERS FILE\n")

    # Training parameters
    file.write("\nTraining parameters:")
    file.write("\nN_E:" + str(N_E) +"\nN_CV:" + str(N_CV) + "\nN_T:" + str(N_T) + "\nN_B:" + str(N_B) + "\nN_Epochs:" + str(N_Epochs) + 
               "\nLR:" + str(lr) + "\nWD:" + str(wd) + "\nT_train:" + str(T_train) + "\nT_test:" + str(T_test) + "\nGRU number:" + str(nGRU) + "\nDataset:" + dataset_file
               + "\nWeights loss: " + str(weights.T))

    # Date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    file.write("\nSimulation Date & Time:"+ dt_string+"\n\n\n")

    file.close()
    print("parameters.txt successfully saved.")

def generate_results_file(loss_array_lin, loss_array_dB, loss_avg, results_path):

    file = open(results_path+"results.txt","w")
    # Title
    file.write("TEST RESULTS FILE\n")

    # Results
    file.write("ax MSE:"    + str(loss_array_lin[0].item()) +"[lin], "+ str(loss_array_dB[0].item()) +" [dB]"+
               "\nay MSE:"  + str(loss_array_lin[1].item()) +"[lin], "+ str(loss_array_dB[1].item()) +" [dB]"+
               "\ndyaw MSE:"+ str(loss_array_lin[2].item()) +"[lin], "+ str(loss_array_dB[2].item()) +" [dB]"+
               "\nvx MSE:"  + str(loss_array_lin[3].item()) +"[lin], "+ str(loss_array_dB[3].item()) +" [dB]"+
               "\nvy MSE:"  + str(loss_array_lin[4].item()) +"[lin], "+ str(loss_array_dB[4].item()) +" [dB]")

    file.write("\n\nAvg MSE:"+ str(loss_avg.item())+ " [dB]")

    # Date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    file.write("\nSimulation Date & Time:"+ dt_string+"\n\n\n")

    file.close()
    print("results.txt successfully saved.")