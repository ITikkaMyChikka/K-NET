import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
from datetime import datetime

# Imports from Kalman_Net_TSP
from KalmanNet_sysmdl import SystemModel
from Pipeline_KF import Pipeline_KF
from KalmanNet_build import NNBuild
from KalmanNet_train import NNTrain
from KalmanNet_test import NNTest
from KalmanNet_nn import KalmanNetNN

# Imports from Adria_KN
import pipeline


# RUNNING ON GPU IF POSSIBLE
if torch.cuda.is_available():
   cuda0 = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
   torch.set_default_tensor_type('torch.cuda.FloatTensor')
   print("Running on the GPU")
else:
   cuda0 = torch.device("cpu")
   print("Running on the CPU")



######################################### START ##############################################


################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)


#############
### PATHS ###
#############

results_path = "/Users/sidhu/Documents/ETH/Semester Project/K-NET/src/Naive_VE/Sim1/Results/"
plots_path = "/Users/sidhu/Documents/ETH/Semester Project/K-NET/src/Naive_VE/Sim1/Plots/"
data_path = "/Datasets/"
dataset_file = "Dataset_50.pickle"
datasets_combined = ["Dataset_20.pickle","Dataset_50.pickle","Dataset_80.pickle","Dataset_200.pickle"]


####################
### LOADING DATA ###
####################
print("Loading data...")
file = "/Users/sidhu/Documents/ETH/Semester Project/Adria_KN/utils_data/Datasets/Dataset_50.pickle"
[train_input, cv_input, test_input,
 train_target, cv_target, test_target,
 train_init, cv_init, test_init,
 T_train, T_test, test_time, rtk_test] = pipeline.load_dataset(file)

train_input = train_input.to(cuda0,non_blocking = True)
cv_input = cv_input.to(cuda0,non_blocking = True)
test_input = test_input.to(cuda0,non_blocking = True)

train_target = train_target.to(cuda0,non_blocking = True)
cv_target = cv_target.to(cuda0,non_blocking = True)
test_target = test_target.to(cuda0,non_blocking = True)

train_init = train_init.to(cuda0,non_blocking = True)
cv_init = cv_init.to(cuda0,non_blocking = True)
test_init = test_init.to(cuda0,non_blocking = True)
print("data loaded successfully")


##########################
### BUILD SYSTEM MODEL ###
##########################

print("Building System Models...")

# Variables describing system model dimensions
obs_dim = 5     # [v_x, v_y, yaw_rate, a_x, a_y]
state_dim = 5   # [v_x, v_y, yaw_rate, a_x, a_y]

# Noise Matrices
Q = torch.eye(state_dim) # TODO: Change this to constant velocity [[00000],[00000],[00100], [00010], [00001]]
R = torch.eye(obs_dim)

# Trajectories
T = cv_input.size(2)
T_test = test_input.size()[2]

# Initialize the System Model
# TODO: Change f_function and h_function to the 5D-State Space Model
sys_model = SystemModel(f = pipeline.f_function,    # State Evolution Model
                        Q = Q,                      # Process Noise
                        h = pipeline.h_function,    # Observation Model
                        R = R,                      # Observation Noise
                        T = T,                      # Trajectory Length for training
                        T_test = T_test,            # Trajectory Length for testing
                        prior_Q = None,             # prior_Q
                        prior_Sigma = None,         # prior_Sigma
                        prior_S = None)             # Prior S

# Initialize the Sequence
m1x_0 = torch.zeros((state_dim,1)).to(cuda0,non_blocking = True) # TODO: keep in mind how Adria does this: state = [ax,ay,dyaw,vx,vy]
m2x_0 = torch.zeros_like(Q).to(cuda0,non_blocking = True) # Initial Covariance Matrix
sys_model.InitSequence(m1x_0, m2x_0)

print("System model initialized")


######################
### K-NET PIPELINE ###
######################
print("Pipeline Start")

# Initialize Pipeline_EKF
KNet_Pipeline = Pipeline_KF(Time = strTime, folderName= "Saved_here", modelName= "KalmanNet")

# Give the System Model to the Pipeline
KNet_Pipeline.setssModel(sys_model)

# Initialize the KalmanNet
KNet_model = KalmanNetNN()

# Building the KalmanNet
KNet_model.Build(sys_model)

# Give the KalmanNet to the Pipeline
KNet_Pipeline.setModel(KNet_model)

# Set training parameters for KalmanNet
KNet_Pipeline.setTrainingParams(n_Epochs=500, n_Batch=100, learningRate=5e-3, weightDecay=1e-4)

# TRAINING
KNet_Pipeline.NNTrain(N_E, train_input, train_target, N_CV, cv_input, cv_target)

# TESTING
[KNet_MSE_test_linear_arr,
 KNet_MSE_test_linear_avg,
 KNet_MSE_test_dB_avg,
 KNet_test] = KNet_Pipeline.NNTest(N_T, test_input, test_target)

# Save Pipeline
KNet_Pipeline.save()

# TODO: Implement plotting of results and saving them


################################################# END #####################################################


