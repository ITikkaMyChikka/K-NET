import torch
import pipeline
from KalmanNet_sysmdl import SystemModel
from KalmanNet_build import NNBuild
from KalmanNet_train import NNTrain
from KalmanNet_test import NNTest
from utils import velocity_integration, geodetic_transform
import KalmanNet_plt as knet_plt
import NN_parameters
import resultsGen
torch.set_printoptions(threshold=10_000)

## Misc stuff
global st
st = -10
obs_dim = 9
state_dim = 5

if torch.cuda.is_available():
  cuda0 = torch.device("cuda:0") # you can continue going on here, like cuda:1 cuda:2....etc. 
  torch.set_default_tensor_type('torch.cuda.FloatTensor')
  print ( "GPU" )
else :
  cuda0 = torch.device("cpu") 
  print( "CPU" )

## Paths to folders
"""
results_path = "Results/Simulation_1/Results/"
plots_path = "Results/Simulation_1/Plots/"
"""
results_path = "/Users/sidhu/Documents/ETH/Semester Project/K-NET/src/Sim1/Results/"
plots_path = "/Users/sidhu/Documents/ETH/Semester Project/K-NET/src/Sim1/Plots/"
data_path = "/Datasets/"
  #"utils_data/Datasets/"
warm_start = False
dynamic_training = False
dataset_file = "Dataset_50.pickle"
datasets_combined = ["Dataset_20.pickle","Dataset_50.pickle","Dataset_80.pickle","Dataset_200.pickle"]

## Load Dataset and convert to GPU
if(dynamic_training):
  dataset_file = datasets_combined
  file_number = len(dataset_file)
  [train_input, cv_input,test_input, train_target, cv_target, test_target, train_init, cv_init, test_init, T_train, T_test, test_time, rtk_test] = pipeline.load_combined_dataset(dataset_file)
  i = 0
  for set in train_input:
    train_input[i] = train_input[i].to(cuda0,non_blocking = True)
    cv_input[i] = cv_input[i].to(cuda0,non_blocking = True)
    test_input[i] = test_input[i].to(cuda0,non_blocking = True)

    train_target[i] = train_target[i].to(cuda0,non_blocking = True)
    cv_target[i] = cv_target[i].to(cuda0,non_blocking = True)
    test_target[i] = test_target[i].to(cuda0,non_blocking = True)

    train_init[i] = train_init[i].to(cuda0,non_blocking = True)
    cv_init[i] = cv_init[i].to(cuda0,non_blocking = True)
    test_init[i] = test_init[i].to(cuda0,non_blocking = True)

else:
  file_number = 0
  file = "/Users/sidhu/Documents/ETH/Semester Project/Adria_KN/utils_data/Datasets/Dataset_50.pickle"
  [train_input, cv_input,test_input, train_target, cv_target, test_target, train_init, cv_init, test_init, T_train, T_test, test_time, rtk_test] = pipeline.load_dataset(file) #data_path+dataset_file

  train_input = train_input.to(cuda0,non_blocking = True)
  cv_input = cv_input.to(cuda0,non_blocking = True)
  test_input = test_input.to(cuda0,non_blocking = True)

  train_target = train_target.to(cuda0,non_blocking = True)
  cv_target = cv_target.to(cuda0,non_blocking = True)
  test_target = test_target.to(cuda0,non_blocking = True)

  train_init = train_init.to(cuda0,non_blocking = True)
  cv_init = cv_init.to(cuda0,non_blocking = True)
  test_init = test_init.to(cuda0,non_blocking = True)

############################
#### Build System Model ####
############################
print("Building System Models...")

## Build Training SystemModel
Q_train = torch.eye(state_dim)
R_train = torch.eye(obs_dim)
m1x_0_train = torch.zeros((state_dim,1)).to(cuda0,non_blocking = True) # state = [ax,ay,dyaw,vx,vy]
m2x_0_train = torch.zeros_like(Q_train).to(cuda0,non_blocking = True)

sysModel_training = SystemModel(pipeline.f_function, Q_train, pipeline.h_function, R_train)
sysModel_training.InitSequence(m1x_0_train, m2x_0_train)

## Build Test SystemModel
Q_test = torch.eye(state_dim)
R_test = torch.eye(obs_dim)
m1x_0_test = torch.zeros((state_dim,1)).to(cuda0,non_blocking = True)
m2x_0_test = torch.zeros_like(Q_test).to(cuda0,non_blocking = True)

sysModel_test = SystemModel(pipeline.f_function, Q_test, pipeline.h_function, R_test)
sysModel_test.InitSequence(m1x_0_test, m2x_0_test)

#######################
###### KalmanNet ######
#######################

## Build KalmanNet
print("Building KalmanNet...")
training_kNet = NNBuild(sysModel_training, pipeline.preprocess_function, NN_parameters.nGRU)
if(warm_start==True):
  training_kNet = torch.load(results_path+'best-model.pt')
else:
  test_KNet = NNBuild(sysModel_test, pipeline.preprocess_function, NN_parameters.nGRU)

## Train KalmanNet
print("Training KalmanNet...")
[MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = NNTrain(sysModel_training,
                                                                                             training_kNet,
                                                                                             cv_input, # Sensor data for Cross Validation
                                                                                             cv_target,# Velocity GT for Cross Validation
                                                                                             train_input,# Sensor data for training
                                                                                             train_target,# Velocity GT for training
                                                                                             train_init, # Initial values for training (v_0)
                                                                                             cv_init, # Initial values for cross validation (v_0)
                                                                                             NN_parameters.N_Epochs, # Epochs
                                                                                             NN_parameters.N_B, # Batches
                                                                                             NN_parameters.learning_rate,
                                                                                             NN_parameters.wd, # L2 Weight Regularization - Weight Decay
                                                                                             results_path,
                                                                                             file_number)

# Save training results
# TODO: cannot save the results
torch.save({"MSE_cv_linear_epoch":MSE_cv_linear_epoch,"MSE_cv_dB_epoch":MSE_cv_dB_epoch,"MSE_train_linear_epoch":MSE_train_linear_epoch, 
            "MSE_train_dB_epoch":MSE_train_dB_epoch},results_path+"train_results.pt")

## Test KalmanNet
print("Testing Model...")
[MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, MSE_test_linear_dim_avg, MSE_test_dB_dim_avg, KGain_array, x_out_array, y_processed,frequencies_test] = NNTest(sysModel_test, test_input, test_target, test_init, results_path)
# Save test results
torch.save({"MSE_test_linear_arr":MSE_test_linear_arr,"MSE_test_linear_avg":MSE_test_linear_avg,
            "MSE_test_dB_avg":MSE_test_dB_avg, "KGain_array":KGain_array, "x_out_array":x_out_array, 
            "y_processed":y_processed, "frequencies":frequencies_test},results_path+"test_results.pt")
print(x_out_array.size())
##############################
### Results Postprocessing ###
##############################
N_E = train_input.size()[0]
N_CV = cv_input.size()[0]
N_T = test_input.size()[0]
plot_limit = round(T_test)
plot_start = round(0)

# Transform RTK to ENU location
lat0 = torch.mean(rtk_test[0,1,0:10])
lon0 = torch.mean(rtk_test[0,0,0:10])
pos_RTK = geodetic_transform(rtk_test[0,:,:],lat0,lon0)

## Integrate velocities to get position
pos_KNet = velocity_integration(x_out_array[0,:,:].detach(), test_time[0,0,:])
pos_MKF = velocity_integration(test_target[0,:,:].detach(), test_time[0,0,:])
pos_integrated = {'KNet':pos_KNet.detach()[:,plot_start:plot_limit], 'MKF':pos_MKF.detach()[:,plot_start:plot_limit], 'RTK':pos_RTK.detach()[:,plot_start:plot_limit]}

## Save results and parameters
resultsGen.generate_params_file(N_E, N_CV, NN_parameters.N_B, N_T, NN_parameters.N_Epochs,NN_parameters.learning_rate,
                                NN_parameters.wd, T_train, T_test, NN_parameters.nGRU, NN_parameters.weights, dataset_file, results_path)
resultsGen.generate_results_file(MSE_test_linear_dim_avg, MSE_test_dB_dim_avg, MSE_test_dB_avg, results_path)

knet_plt.plotTrajectories(x_out_array[0,:,plot_start:plot_limit].detach(), test_target[0,:,plot_start:plot_limit].detach(), test_time[0,0,plot_start:plot_limit].detach(), 
                          y_processed[0,:,plot_start:plot_limit], pos_integrated, plots_path + 'trajectories_0.png')
knet_plt.NNPlot_train(MSE_test_linear_arr, MSE_test_dB_avg,MSE_cv_dB_epoch, MSE_train_dB_epoch,plots_path)

