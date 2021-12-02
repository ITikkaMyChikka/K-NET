
import pickle
import torch
import models
import matplotlib.pyplot as plt

import sys
sys.path.insert(1, 'Logdata/')

from utils_data import Dataset

def load_dataset(file_name):
    with open(file_name, "rb") as f:
        print(f)
        my_dataset = pickle.load(f)
    print("Loading ", my_dataset.name, "...")
    train_data = my_dataset.train_data
    test_data = my_dataset.test_data
    cv_data = my_dataset.cv_data

    train_input = format_input(train_data)
    cv_input = format_input(cv_data)
    test_input = format_input(test_data)

    train_target = train_data["VE"].float()
    cv_target = cv_data["VE"].float()
    test_target = test_data["VE"].float()

    train_init = train_data["Initial_VE"].float()
    cv_init = cv_data["Initial_VE"].float()
    test_init = test_data["Initial_VE"].float()

    test_time = test_data["Time"].float()
    rtk_test = test_data["RTK"].float()

    print(my_dataset.name, "successfully loaded")
    return [train_input, cv_input, test_input, train_target, cv_target, test_target, train_init, cv_init, test_init, my_dataset.T_train, my_dataset.T_test, test_time, rtk_test]

def load_combined_dataset(file_array):

    train_input_sets = []
    val_input_sets = []
    test_input_sets = []

    train_target_sets = []
    val_target_sets = []
    test_target_sets = []

    train_init_sets = []
    val_init_sets = []
    test_init_sets = []

    T_train_sets = []
    T_test_sets = []
    test_time_sets = []
    rtk_test_sets = []

    for file in file_array:
        [train_input, cv_input, test_input, train_target, cv_target, test_target, train_init, cv_init, test_init, T_train, T_test, test_time, rtk_test] = load_dataset("Datasets/"+file)
        train_input_sets.append(train_input)
        val_input_sets.append(cv_input)
        test_input_sets.append(test_input)
        train_target_sets.append(train_target)
        val_target_sets.append(cv_target)
        test_target_sets.append(test_target)
        train_init_sets.append(train_init)
        val_init_sets.append(cv_init)
        test_init_sets.append(test_init)
        T_train_sets.append(T_train)
        T_test_sets.append(T_test)
        test_time_sets.append(test_time)
        rtk_test_sets.append(rtk_test)
    
    return [train_input_sets,val_input_sets,test_input_sets,train_target_sets,val_target_sets,test_target_sets,train_init_sets,val_init_sets,test_init_sets,T_train_sets,T_test_sets,test_time_sets,rtk_test_sets]



def format_input(data_dic):
    ## In this implementation we assume that measurement is the following:
    # [ax_imu,ay_imu,az_imu,dyaw_imu,ax_ins,ay_ins,az_ins,dyaw_ins,rpm_rl,rpm_rr,rpm_fl,rpm_fr,tm_r,tm_l,sa]
    data_tens = torch.cat((data_dic["IMU"][:,[0,1,2,5],:],data_dic["INS"][:,[0,1,2,5],:],data_dic["RPM"],data_dic["MT"],data_dic["SA"]),dim=1)
    return data_tens.float()


def f_function(state):
    ## State:
    # [ax,ay,dyaw,vx,vy]
    return models.motion(state.detach())

def preprocess_function(state, measurement):
    ## In this implementation we assume that measurement is the following:
    # [ax_imu(0),ay_imu(1),az_imu(2),dyaw_imu(3),ax_ins(4),ay_ins(5),az_ins(6),dyaw_ins(7),rpm_rl(8),rpm_rr(9),rpm_fl(10),rpm_fr(11),4xtm(12,13,14,15),sa(16)]

    # Vehicle model
    # Obs: [ax_imu, ay_imu, dyaw_imu, ax_ins, ay_ins, dyaw_ins, vx_rpm, vy_rpm, dyaw_rpm]
    vehicle_model = models.VD_Mean_Model(state, measurement[16,:])
    y_imu = models.model_imu(measurement[0:4,:])
    y_ins = models.model_ins(measurement[4:8,:])
    # For input reduction
    y_acc = (0.5*y_imu + 0.5*y_ins)
    y_ws = models.model_rpm(measurement[8:12,:], vehicle_model)[:,:]
    y = torch.cat((y_imu,y_ins,y_ws),dim=0)
    #y = torch.cat((y_acc,y_ws),dim=0)
    # Set NaN to 0
    y[torch.isnan(y)] = 0
    return y
    
def h_function(state):
    ## State:
    # [ax,ay,dyaw,vx,vy]

    # Vehicle model
    # Obs: [ax_imu, ay_imu, dyaw_imu, ax_ins, ay_ins, dyaw_ins, vx_rpm, vy_rpm]
    y_imu = models.h_imu(state)
    y_ins = models.h_ins(state)
    y_ws = models.h_rpm(state)[:,:]

    y_acc = y_imu

    y_pred = torch.cat((y_imu,y_ins,y_ws),dim=0)
    #y_pred = torch.cat((y_acc,y_ws),dim=0)
    return y_pred


