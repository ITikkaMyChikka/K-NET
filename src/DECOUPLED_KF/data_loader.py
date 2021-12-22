import pickle
import torch

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

def format_input(data_dic):
    ## In this implementation we assume that measurement is the following:
    # [ax_imu,ay_imu,az_imu,dyaw_imu,ax_ins,ay_ins,az_ins,dyaw_ins,rpm_rl,rpm_rr,rpm_fl,rpm_fr,tm_r,tm_l,sa]
    data_tens = torch.cat((data_dic["IMU"][:,[0,1,2,5],:],data_dic["INS"][:,[0,1,2,5],:],data_dic["RPM"],data_dic["MT"],data_dic["SA"]),dim=1)
    return data_tens.float()
