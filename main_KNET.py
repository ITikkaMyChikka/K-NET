###############
### IMPORTS ###
###############
# Generic Imports
from datetime import datetime
import torch
import os
# KalmanNet_TSP imports
from src.KalmanNet_TSP import KalmanNet_sysmdl
from src.KalmanNet_TSP import EKF_test
from src.KalmanNet_TSP import Extended_KalmanNet_nn
# My imports
from src.motion_models import motion_model_1

from data_generator import DataGen
from visualizer import visualizer
from amz_dataloader import load_dataset
from Pipeline_KNet import Pipeline_KNET
from data_generator import N_E, N_CV, N_T

############
### CUDA ###
############

if torch.cuda.is_available():
    cuda0 = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("Running on the GPU")
else:
    cuda0 = torch.device("cpu")
    print("Running on the CPU")

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
# Paths to folders
results_path = "/Users/sidhu/Documents/ETH/Semester Project/Adria_KN/src/Sim1/Results/"
plots_path = "/Results/"
data_path = "/Datasets/"
dataset_file = "Dataset_50.pickle"
datasets_combined = ["Dataset_20.pickle", "Dataset_50.pickle", "Dataset_80.pickle", "Dataset_200.pickle"]
DatafolderName = 'SyntheticData/EKFNet_paper/'


########################
### LOADING AMZ DATA ###
########################
file_number = 0
file = "/Users/sidhu/Documents/ETH/Semester Project/Adria_KN/utils_data/Datasets/Dataset_50.pickle"
[train_input_AMZ, cv_input_AMZ,test_input_AMZ, train_target_AMZ, cv_target_AMZ, test_target_AMZ, train_init_AMZ, cv_init_AMZ, test_init_AMZ, T_train_AMZ, T_test_AMZ, test_time_AMZ, rtk_test_AMZ] = load_dataset(file) #data_path+dataset_file

train_input_AMZ = train_input_AMZ.to(cuda0,non_blocking = True)
cv_input_AMZ = cv_input_AMZ.to(cuda0,non_blocking = True)
test_input_AMZ = test_input_AMZ.to(cuda0,non_blocking = True)

train_target_AMZ = train_target_AMZ.to(cuda0,non_blocking = True)
cv_target_AMZ = cv_target_AMZ.to(cuda0,non_blocking = True)
test_target_AMZ = test_target_AMZ.to(cuda0,non_blocking = True)

train_init_AMZ = train_init_AMZ.to(cuda0,non_blocking = True)
cv_init_AMZ = cv_init_AMZ.to(cuda0,non_blocking = True)
test_init_AMZ = test_init_AMZ.to(cuda0,non_blocking = True)

"""
This is only for input_AMZ
[ax_imu,ay_imu,az_imu,dyaw_imu,ax_ins,ay_ins,az_ins,dyaw_ins,rpm_rl,rpm_rr,rpm_fl,rpm_fr,tm_r,tm_l,sa]

"""
# TODO: ULTIMATE GOAL: We want to create SyntheticData but with observations from AMZ_dataset
# TODO: Which motion model to use?
# TODO: Implement motion model
# TODO: Generate the synthetic data
# TODO: PreTrain KalmanNet with synthetic data
# TODO: Look up how to save parameters or model
# TODO: Train KalmanNet

# TODO: Just Implement KalmanNet

#######################################
### DEFINE MOTION/STATE SPACE MODEL ###
#######################################
# Define all functions and noise matrices
f = motion_model_1.f
h = motion_model_1.h
Q = motion_model_1.Q
R = motion_model_1.R
# Define Trajectory Length
T_train, T_test = 200, 200  # T-train defines the trajectory length for training&validation dataset

# Initialize SystemModel
sys_model = KalmanNet_sysmdl.SystemModel(f=f, Q=Q, h=h, R=R, T=T_train, T_test=T_test)

# Initialize initial values for state vectors and covariance matrix
m1x_0 = torch.tensor([0.75, 0.5, 0.5, 1.2, 1.1]).float()  # x_0 = [ax, ay, phi, vx, vy]
m2x_0 = 2000 * torch.eye(5, 5).float()  # P_0 = Identity * 2000, to show that we are very uncertain at the beginning
sys_model.InitSequence(m1x_0=m1x_0, m2x_0=m2x_0)

###############################
### GENERATE SYNTHETIC DATA ###
###############################
dataFileName = ['syntheticData_KNET_AMZ.pt']  # Naming the dataset

# We only want to generate data if the folder is empty
if os.path.isfile("./" + dataFileName[0]):
    print("Start Data Gen")
    # Generating data
    DataGen(sys_model, DatafolderName + dataFileName[0], T_train, T_test)

##############################
### LOADING SYNTHETIC DATA ###
##############################
print("Data Load")
print(dataFileName[0])
[train_input, train_target, cv_input, cv_target, test_input, test_target] = torch.load(DatafolderName + dataFileName[0],
                                                                                       map_location=cuda0)
print("trainset size:", train_target.size())
print("cvset size:", cv_target.size())
print("testset size:", test_target.size())



##############################
### EXTENDED KALMAN FILTER ###
##############################
print("Evaluate EKF")
[MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, EKF_KG_array, EKF_out] = EKF_test.EKFTest(sys_model,
                                                                                                   test_input,
                                                                                                   test_target)

"""
MSE_EKF_linear_arr: MS-Error of each item in the trajectory
MSE_EKF_linear_avg: Average MSE
MSE_EKF_dB_avg: Average MSE in dB
EKF_KG_array: Kalman Gain Matrix ?
EKF_out: [ax, ay, phi, vx, vy] all the states at each time step of the trajectory
"""


#################
### KALMANNET ###
#################

KalmanNet_Pipeline = Pipeline_KNET("KNet", "KalmanNet")
KalmanNet_Pipeline.setssModel(sys_model)
KNet_model = Extended_KalmanNet_nn.KalmanNetNN()
KNet_model.Build(sys_model)
KalmanNet_Pipeline.setModel(KNet_model)
KalmanNet_Pipeline.setTrainingParams(n_Epochs=10, n_Batch=10, learningRate=5e-3, weightDecay=1e-4)

N_E_AMZ = train_input_AMZ.size()[0]
N_CV_AMZ = cv_input_AMZ.size()[0]
N_T_AMZ = test_input_AMZ.size()[0]

# KALMAN_NET with SyntheticData
KalmanNet_Pipeline.NNTrain(N_E, train_input, train_target, N_CV, cv_input, cv_target)
[KNet_MSE_test_linear_arr, KNet_MSE_test_linear_avg, KNet_MSE_test_dB_avg, KNet_test] = KalmanNet_Pipeline.NNTest(N_T, test_input, test_target)
KalmanNet_Pipeline.save()

"""
KALMANNET WITH AMZ_DATASET
# TODO: Preprocessing to cv_input_amz = y_cv, such that it matches the motion model observations, also train_input_AMZ
# TODO: Preprocessing to cv_target to be of same size as x_out (state), also train_target_AMZ
KalmanNet_Pipeline.NNTrain(N_E_AMZ, train_input_AMZ, train_target_AMZ, N_CV_AMZ, cv_input_AMZ, cv_target_AMZ)
[KNet_MSE_test_linear_arr, KNet_MSE_test_linear_avg, KNet_MSE_test_dB_avg, KNet_test] = KalmanNet_Pipeline.NNTest(N_T, test_input, test_target)
KalmanNet_Pipeline.save()
"""








################
### PLOTTING ###
################

print("Plotting Results...")
visualizer(KNet_test, EKF_out, test_target, T_test, plots_path)

