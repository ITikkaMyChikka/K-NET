import numpy as np
import torch
import math
import scipy.io
from scipy import signal
from torch.functional import split
import matplotlib.pyplot as plt

file_dictionary = { 
                    # Tuggen 23/08/2020
                    1:'Logdata/20200823_tuggen_rtk/Log_20200823_094407',
                    2:'Logdata/20200823_tuggen_rtk/Log_20200823_102215',
                    3:'Logdata/20200823_tuggen_rtk/Log_20200823_104528',
                    4:'Logdata/20200823_tuggen_rtk/Log_20200823_112842',
                    5:'Logdata/20200823_tuggen_rtk/Log_20200823_113455',
                    6:'Logdata/20200823_tuggen_rtk/Log_20200823_115319',
                    7:'Logdata/20200823_tuggen_rtk/Log_20200823_123534',
                    8:'Logdata/20200823_tuggen_rtk/Log_20200823_133304',
                    9:'Logdata/20200823_tuggen_rtk/Log_20200823_133956',
                    10:'Logdata/20200823_tuggen_rtk/Log_20200823_134642',
                    11:'Logdata/20200823_tuggen_rtk/Log_20200823_135824',
                    12:'Logdata/20200823_tuggen_rtk/Log_20200823_141510',
                    13:'Logdata/20200823_tuggen_rtk/Log_20200823_142706',
                    14:'Logdata/20200823_tuggen_rtk/Log_20200823_162812',
                    15:'Logdata/20200823_tuggen_rtk/Log_20200823_163434',
                    16:'Logdata/20200823_tuggen_rtk/Log_20200823_165526',
                    17:'Logdata/20200823_tuggen_rtk/Log_20200823_170525',
                    18:'Logdata/20200823_tuggen_rtk/Log_20200823_180048',
                    19:'Logdata/20200823_tuggen_rtk/Log_20200823_193310',
                    20:'Logdata/20200823_tuggen_rtk/Log_20200823_193708',
                    # Alpanach 22/08/2020
                    21:'Logdata/20200822_alpnach/Log_20200821_185731'   ,
                    22:'Logdata/20200822_alpnach/Log_20200821_185924'   ,
                    23:'Logdata/20200822_alpnach/Log_20200821_195222'   ,
                    24:'Logdata/20200822_alpnach/Log_20200821_195835'   }

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def gaussian_smoothing(x):
    # Function to smooth the ground truth data via Gaussian smoothing with custom weights
    # Input: tensor in the shape torch.tensor() size 5xT 
    # Create gaussian window
    window_size = 15 # Must be an odd number for dimensionality matching!
    std = 1
    gaussian_weights = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(signal.gaussian(window_size, std=std)),dim=0),dim=0).float()
    gaussian_weights = gaussian_weights/torch.sum(gaussian_weights)
    pad = (torch.tensor(gaussian_weights.size()[2:]) - 1) // 2
    pad = tuple(pad.tolist())
    conv = torch.nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size=window_size, bias=False, padding=pad)
    with torch.no_grad():
        conv.weight = torch.nn.Parameter(gaussian_weights)

    # Apply smoothing
    x_smooth = torch.zeros_like(x)
    for i in range(x.size(1)):
        x_smooth[:,i:i+1,:] = conv(x[:,i:i+1,:].float())
    return x_smooth.float()

def time_finder(time, vx):
    threshold = 0
    bias_buffer = 400
    mask = vx > threshold
    try:
        start_time = np.min(time[mask])
        print(start_time)
        end_time = np.max(time[mask])
        print(end_time)

        start_index = np.maximum(np.where(time == start_time)[0].item() - bias_buffer,0)
        end_index = np.where(time == end_time)[0].item() + bias_buffer
    except:
        start_index = end_index = 0

    # Buffer for bias calibration calculations 
    start_index = max(0,start_index - 500)

    return[start_index,end_index]

def read_vars(sens, start, end):
    '''
    All necessary data is obtained through this function
    '''

    #############
    ## Pilatus ##
    #############

    # Time
    time = np.squeeze(sens['Time'][0,0])
    TIME_data = np.expand_dims(time,1)

    # IMU
    print ('- Loading IMU Data...')

    IMU_a_x = np.squeeze(sens['a_x'][0,0])
    IMU_a_y = np.squeeze(sens['a_y'][0,0])
    IMU_a_z = np.squeeze(sens['a_z'][0,0])

    IMU_gyro_x = np.squeeze(sens['gyro_x'][0,0])
    IMU_gyro_y = np.squeeze(sens['gyro_y'][0,0])
    IMU_gyro_z = np.squeeze(sens['gyro_z'][0,0]) 

    IMU_data = np.vstack((IMU_a_x, IMU_a_y, IMU_a_z, IMU_gyro_x, IMU_gyro_y, IMU_gyro_z))

    # INS
    print('- Loading INS Data...')

    try:
        INS_a_x = np.squeeze(sens['INS_acc_x'][0,0])
        INS_a_y = np.squeeze(sens['INS_acc_y'][0,0])
        INS_a_z = np.squeeze(sens['INS_acc_z'][0,0])

        INS_gyro_x = np.squeeze(sens['INS_gyro_x'][0,0])
        INS_gyro_y = np.squeeze(sens['INS_gyro_y'][0,0])
        INS_gyro_z = np.squeeze(sens['INS_gyro_z'][0,0]) 

        INS_data = np.vstack((INS_a_x, INS_a_y, INS_a_z, INS_gyro_x, INS_gyro_y, INS_gyro_z))

    except:
        print('-> No INS Data')
        INS_data = np.zeros_like(IMU_data)

    # GSS
    print('- Loading GSS Data...')

    GSS_v_x = np.squeeze(sens['ASS_vx'][0,0])
    GSS_v_y = np.squeeze(sens['ASS_vy'][0,0])

    GSS_data = np.vstack((GSS_v_x,GSS_v_y))

    # GPD [GPSVx, GPSVy, Head, HeadVal, Lat, Long]
    print('- Loading GPD Data...')

    try:
        GPS_vx = np.squeeze(sens['GPS_antenna_speed'][0,0])
        GPS_vy = np.squeeze(sens['GPS_heading_diff'][0,0])

        head = np.squeeze(sens['GPS_dual_heading'][0,0])
        head_val = np.squeeze(sens['INS_GPD_val'][0,0])
        gps_lat = np.squeeze(sens['GPS_latitude'][0,0]) 
        gps_long = np.squeeze(sens['GPS_longitude'][0,0]) 

        GPD_data = np.vstack((GPS_vx, GPS_vy, head, head_val, gps_lat, gps_long))

    except:
        print('-> No GPD Data')
        GPD_data = np.zeros((6,np.shape(time)[0]))

    # GPS [vx,vy,lat,long]
    print('- Loading GPS Data...')

    GPS_vx = np.squeeze(sens['Vel_N_raw'][0,0])
    GPS_vy = np.squeeze(sens['Vel_E_raw'][0,0])
    GPS_lat = np.squeeze(sens['Latitude_out'][0,0])
    GPS_long = np.squeeze(sens['Longitude_out'][0,0])

    GPS_data = np.vstack((GPS_vx, GPS_vy, GPS_lat, GPS_long))

    # RPM
    print('- Loading RPM Data...')

    motor_FL = np.squeeze(sens['Motor_FL_rpm'][0,0])
    motor_FR = np.squeeze(sens['Motor_FR_rpm'][0,0])
    motor_RL = np.squeeze(sens['Motor_RL_rpm'][0,0])
    motor_RR = np.squeeze(sens['Motor_RR_rpm'][0,0])

    RPM_data = np.vstack((motor_FL, motor_FR, motor_RL, motor_RR))

    # ALP
    print('- Loading ALP Data...')

    FL_drpm = np.diff(motor_FL)
    FR_drpm = np.diff(motor_FR)
    RL_drpm = np.diff(motor_RL)
    RR_drpm = np.diff(motor_RR)

    zeros_aux = np.zeros_like(FL_drpm)

    ALP_data = np.vstack((zeros_aux, FL_drpm, zeros_aux, FR_drpm, zeros_aux, RL_drpm, zeros_aux, RR_drpm))

    # Motor Torques
    print('- Loading MT Data...')
    conv = 17/120 # Motor current to torque conversion

    torque_FL = conv * np.squeeze(sens['Motor_FL_currentq'][0,0])
    torque_FR = conv * np.squeeze(sens['Motor_FR_currentq'][0,0])
    torque_RL = conv * np.squeeze(sens['Motor_RL_currentq'][0,0])
    torque_RR = conv * np.squeeze(sens['Motor_RR_currentq'][0,0])

    MT_data = np.vstack((torque_FL, torque_FR, torque_RL, torque_RR))

    # Steering angle
    print('- Loading SA Data...')
    steering_angle = np.squeeze(sens['Steering_angle_out'][0,0])

    SA_data = np.expand_dims(steering_angle,0)

    # Driver command
    DC_data = np.squeeze(sens['GP_out'][0,0]) - np.squeeze(sens['BP_out'][0,0])
    DC_data = np.expand_dims(DC_data,0)

    # Online VE
    print('- Loading Online VE...')

    VE_ax = sens['VE_ax'][0,0]
    VE_ay = sens['VE_ay'][0,0]
    VE_dy = sens['VE_dy'][0,0]
    VE_vx = sens['VE_vx'][0,0]
    VE_vy = sens['VE_vy'][0,0]

    VE_data = np.hstack((VE_ax,VE_ay,VE_dy,VE_vx,VE_vy))
    # Get rid of nans by interpolation
    for dim in range(VE_data.shape[1]):
        y = VE_data[:,dim]
        nans, x= nan_helper(y)
        y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        VE_data[:,dim] = y

    # RTK
    print('- Loading RTK...')
    RTK_long = sens['RTK_longitude'][0,0]
    RTK_lat  = sens['RTK_latitude'][0,0]

    RTK_data = np.hstack((RTK_long,RTK_lat))

    return [TIME_data[start:end,:], IMU_data.T[start:end,:], INS_data.T[start:end,:], GSS_data.T[start:end,:], GPD_data.T[start:end,:],GPS_data.T[start:end,:], 
            RPM_data.T[start:end,:], ALP_data.T[start:end,:], MT_data.T[start:end,:], SA_data.T[start:end,:], DC_data.T[start:end,:], VE_data[start:end,:], RTK_data[start:end,:]]

def read_mat(file_path):
    f = scipy.io.loadmat(file_path+'.mat')

    # Find the interesting time slot
    vx = np.squeeze(f['sens']['VE_vx'][0,0])
    time = np.squeeze(f['sens']['Time'][0,0])
    [start, end] = time_finder(time, vx)
    print("Start Index:", start," End Index:", end, " Total length:",end-start)

    if(end-start <= 100):
        print("File ", file_path, "doesn't go over speed threshold")
        return False

    # Load all variables of interest
    sens = f['sens']
    [time, IMU_data, INS_data, GSS_data, GPD_data, GPS_data, RPM_data, ALP_data, MT_data, SA_data, DC_data, VE_data, RTK_data] = read_vars(sens, start, end)

    # Convert tensors to torch
    time_torch = torch.from_numpy(time).T
    IMU_torch  = torch.from_numpy(IMU_data).T
    INS_torch  = torch.from_numpy(INS_data).T
    GSS_torch  = torch.from_numpy(GSS_data).T
    GPD_torch  = torch.from_numpy(GPD_data).T
    GPS_torch  = torch.from_numpy(GPS_data).T
    RPM_torch  = torch.from_numpy(RPM_data).T
    ALP_torch  = torch.from_numpy(ALP_data).T
    MT_torch   = torch.from_numpy(MT_data).T
    SA_torch   = torch.from_numpy(SA_data).T
    DC_torch   = torch.from_numpy(DC_data).T
    VE_torch   = torch.from_numpy(VE_data).T
    RTK_torch  = torch.from_numpy(RTK_data).T

    # Smooth Ground Truth Data via Gaussian non-causal average (Have to do unsqueeze and squeeze because of stupid torch)
    VE_torch = torch.squeeze(gaussian_smoothing(torch.unsqueeze(VE_torch,dim=0)),dim=0)

    torch.save({"Time":time_torch, "IMU":IMU_torch, "INS":INS_torch, "GSS":GSS_torch, "GPD":GPD_torch, "GPS":GPS_torch, 
                "RPM":RPM_torch, "ALP":ALP_torch, "MT":MT_torch, "SA":SA_torch, "DC":DC_torch, "VE":VE_torch, "RTK":RTK_torch}, file_path+'.pt')

    print("Data load of ",file_path, " has been successful")
    return True

class Dataset:
    def __init__(self, name):
        self.name = name
        imu_angle = [0.0015,0.0538,-3.3671]
        ins_angle = [-0.7187,0.6865,-1.4085]

        self.C_imu = torch.mm(torch.mm(rotate_ang(imu_angle[2], 3),rotate_ang(imu_angle[1], 2)),rotate_ang(imu_angle[0], 1))
        self.C_ins = torch.mm(torch.mm(rotate_ang(ins_angle[2], 3),rotate_ang(ins_angle[1], 2)),rotate_ang(ins_angle[0], 1))
    
    def create_training_set(self, file_array, T):
        ## Inputs:
        # file_array: names of all the *.pt files that are going into the dataset
        # T: length of each training sequence (in time steps)
        
        self.T_train = T
        self.train_files = file_array

        # Initiallize dataset bags
        time_bag = torch.empty(0,1,self.T_train)
        imu_bag = torch.empty(0,6,self.T_train)
        ins_bag = torch.empty(0,6,self.T_train)
        gss_bag = torch.empty(0,2,self.T_train)
        gpd_bag = torch.empty(0,6,self.T_train)
        gps_bag = torch.empty(0,4,self.T_train)
        rpm_bag = torch.empty(0,4,self.T_train)
        alp_bag = torch.empty(0,8,self.T_train)
        mt_bag = torch.empty(0,4,self.T_train)
        sa_bag = torch.empty(0,1,self.T_train)
        
        ve_bag = torch.zeros(0,5,self.T_train)
        initial_ve_bag = torch.zeros(0,5)

        for file_name in file_array:
            data = torch.load(file_name+'.pt')

            bias_imu = self.compute_bias(data["IMU"],'imu')
            bias_ins = self.compute_bias(data["INS"],'ins')

            data["IMU"] = data["IMU"] - bias_imu
            data["INS"] = data["INS"] - bias_ins

            time_bag = torch.cat((time_bag, self.reshape_tensor(data["Time"], self.T_train)),0)
            imu_bag = torch.cat((imu_bag, self.reshape_tensor(data["IMU"], self.T_train)),0)
            ins_bag = torch.cat((ins_bag, self.reshape_tensor(data["INS"], self.T_train)),0)
            gss_bag = torch.cat((gss_bag, self.reshape_tensor(data["GSS"], self.T_train)),0)
            gpd_bag = torch.cat((gpd_bag, self.reshape_tensor(data["GPD"], self.T_train)),0)
            gps_bag = torch.cat((gps_bag, self.reshape_tensor(data["GPS"], self.T_train)),0)
            rpm_bag = torch.cat((rpm_bag, self.reshape_tensor(data["RPM"], self.T_train)),0)
            alp_bag = torch.cat((alp_bag, self.reshape_tensor(data["ALP"], self.T_train)),0)
            mt_bag = torch.cat((mt_bag, self.reshape_tensor(data["MT"], self.T_train)),0)
            sa_bag = torch.cat((sa_bag, self.reshape_tensor(data["SA"], self.T_train)),0)
            ve_bag = torch.cat((ve_bag, self.reshape_tensor(data["VE"], self.T_train)),0)

        # We need the initial VE in order to initialize KalmanNet at each batch
        initial_ve_bag = ve_bag[:,:,0]

        # Shuffle tensors
        perm = torch.randperm(time_bag.size()[0]) # Random permutation

        time_bag = time_bag[perm,:,:]
        imu_bag = imu_bag[perm,:,:]
        ins_bag = ins_bag[perm,:,:]
        gss_bag = gss_bag[perm,:,:]
        rpm_bag = rpm_bag[perm,:,:]
        alp_bag = alp_bag[perm,:,:]
        mt_bag = mt_bag[perm,:,:]
        sa_bag = sa_bag[perm,:,:]
        ve_bag = ve_bag[perm,:,:]
        initial_ve_bag = initial_ve_bag[perm,:]

        #Set NaNs to 0
        time_bag[torch.isnan(time_bag)] = 0
        imu_bag[torch.isnan(imu_bag)] = 0
        ins_bag[torch.isnan(ins_bag)] = 0
        gss_bag[torch.isnan(gss_bag)] = 0
        rpm_bag[torch.isnan(rpm_bag)] = 0
        alp_bag[torch.isnan(alp_bag)] = 0
        mt_bag[torch.isnan(mt_bag)] = 0
        sa_bag[torch.isnan(sa_bag)] = 0
        ve_bag[torch.isnan(ve_bag)] = 0
        initial_ve_bag[torch.isnan(initial_ve_bag)] = 0

        self.train_data = {"Time":time_bag, "IMU":imu_bag, "INS":ins_bag, "GSS":gss_bag, "GPD":gpd_bag, "GPS":gps_bag, "RPM":rpm_bag,
                    "ALP":alp_bag, "MT":mt_bag, "SA":sa_bag, "VE":ve_bag, "Initial_VE":initial_ve_bag}

    def create_validation_set(self, file_array, N_CV):
        ## Inputs:
        # file_array: names of all the *.pt files that are going into the dataset
        # T: length of each training sequence (in time steps)
        
        self.T_cv = self.T_train
        self.cv_files = file_array

        # Initiallize dataset bags
        time_bag = torch.empty(0,1,self.T_cv)
        imu_bag = torch.empty(0,6,self.T_cv)
        ins_bag = torch.empty(0,6,self.T_cv)
        gss_bag = torch.empty(0,2,self.T_cv)
        gpd_bag = torch.empty(0,6,self.T_cv)
        gps_bag = torch.empty(0,4,self.T_cv)
        rpm_bag = torch.empty(0,4,self.T_cv)
        alp_bag = torch.empty(0,8,self.T_cv)
        mt_bag = torch.empty(0,4,self.T_cv)
        sa_bag = torch.empty(0,1,self.T_cv)
        
        ve_bag = torch.zeros(0,5,self.T_cv)
        initial_ve_bag = torch.zeros(0,5)

        for file_name in file_array:
            data = torch.load(file_name+'.pt')

            bias_imu = self.compute_bias(data["IMU"],'imu')
            bias_ins = self.compute_bias(data["INS"],'ins')

            data["IMU"] = data["IMU"] - bias_imu
            data["INS"] = data["INS"] - bias_ins

            time_bag = torch.cat((time_bag, self.reshape_tensor(data["Time"], self.T_cv)),0)
            imu_bag = torch.cat((imu_bag, self.reshape_tensor(data["IMU"], self.T_cv)),0)
            ins_bag = torch.cat((ins_bag, self.reshape_tensor(data["INS"], self.T_cv)),0)
            gss_bag = torch.cat((gss_bag, self.reshape_tensor(data["GSS"], self.T_cv)),0)
            gpd_bag = torch.cat((gpd_bag, self.reshape_tensor(data["GPD"], self.T_cv)),0)
            gps_bag = torch.cat((gps_bag, self.reshape_tensor(data["GPS"], self.T_cv)),0)
            rpm_bag = torch.cat((rpm_bag, self.reshape_tensor(data["RPM"], self.T_cv)),0)
            alp_bag = torch.cat((alp_bag, self.reshape_tensor(data["ALP"], self.T_cv)),0)
            mt_bag = torch.cat((mt_bag, self.reshape_tensor(data["MT"], self.T_cv)),0)
            sa_bag = torch.cat((sa_bag, self.reshape_tensor(data["SA"], self.T_cv)),0)
            ve_bag = torch.cat((ve_bag, self.reshape_tensor(data["VE"], self.T_cv)),0)

        # We need the initial VE in order to initialize KalmanNet at each batch
        initial_ve_bag = ve_bag[:,:,0]

        # Shuffle tensors
        perm = torch.randperm(time_bag.size(0))[:N_CV] # Random permutation

        time_bag = time_bag[perm,:,:]
        imu_bag = imu_bag[perm,:,:]
        ins_bag = ins_bag[perm,:,:]
        gss_bag = gss_bag[perm,:,:]
        rpm_bag = rpm_bag[perm,:,:]
        alp_bag = alp_bag[perm,:,:]
        mt_bag = mt_bag[perm,:,:]
        sa_bag = sa_bag[perm,:,:]
        ve_bag = ve_bag[perm,:,:]
        initial_ve_bag = initial_ve_bag[perm,:]

        #Set NaNs to 0
        time_bag[torch.isnan(time_bag)] = 0
        imu_bag[torch.isnan(imu_bag)] = 0
        ins_bag[torch.isnan(ins_bag)] = 0
        gss_bag[torch.isnan(gss_bag)] = 0
        rpm_bag[torch.isnan(rpm_bag)] = 0
        alp_bag[torch.isnan(alp_bag)] = 0
        mt_bag[torch.isnan(mt_bag)] = 0
        sa_bag[torch.isnan(sa_bag)] = 0
        ve_bag[torch.isnan(ve_bag)] = 0
        initial_ve_bag[torch.isnan(initial_ve_bag)] = 0


        self.cv_data = {"Time":time_bag, "IMU":imu_bag, "INS":ins_bag, "GSS":gss_bag, "GPD":gpd_bag, "GPS":gps_bag, "RPM":rpm_bag,
                    "ALP":alp_bag, "MT":mt_bag, "SA":sa_bag, "VE":ve_bag, "Initial_VE":initial_ve_bag}

    def create_test_set(self, test_file_array, T=0, N_T=0):
        ## Inputs:
        # test_file: name of the file we are gonna use to test

        # Time length of each of the test examples
        if T==0:
            data = torch.load(test_file_array[0]+'.pt')
            self.T_test = data["Time"].size(1)
        else:
            self.T_test = T
    
        self.test_files = test_file_array

        # Initiallize dataset bags
        time_bag = torch.empty(0,1,self.T_test)
        imu_bag = torch.empty(0,6,self.T_test)
        ins_bag = torch.empty(0,6,self.T_test)
        gss_bag = torch.empty(0,2,self.T_test)
        gpd_bag = torch.empty(0,6,self.T_test)
        gps_bag = torch.empty(0,4,self.T_test)
        rpm_bag = torch.empty(0,4,self.T_test)
        alp_bag = torch.empty(0,8,self.T_test)
        mt_bag = torch.empty(0,4,self.T_test)
        sa_bag = torch.empty(0,1,self.T_test)
        rtk_bag = torch.empty(0,2,self.T_test)
        
        ve_bag = torch.zeros(0,5,self.T_test)
        initial_ve_bag = torch.zeros(0,5)

        for test_file in test_file_array:
            data = torch.load(test_file+'.pt')

            bias_imu = self.compute_bias(data["IMU"],'imu')
            bias_ins = self.compute_bias(data["INS"],'ins')

            data["IMU"] = data["IMU"] - bias_imu
            data["INS"] = data["INS"] - bias_ins

            time_bag = torch.cat((time_bag, self.reshape_tensor(data["Time"], self.T_test)),0)
            imu_bag = torch.cat((imu_bag, self.reshape_tensor(data["IMU"], self.T_test)),0)
            ins_bag = torch.cat((ins_bag, self.reshape_tensor(data["INS"], self.T_test)),0)
            gss_bag = torch.cat((gss_bag, self.reshape_tensor(data["GSS"], self.T_test)),0)
            gpd_bag = torch.cat((gpd_bag, self.reshape_tensor(data["GPD"], self.T_test)),0)
            gps_bag = torch.cat((gps_bag, self.reshape_tensor(data["GPS"], self.T_test)),0)
            rpm_bag = torch.cat((rpm_bag, self.reshape_tensor(data["RPM"], self.T_test)),0)
            alp_bag = torch.cat((alp_bag, self.reshape_tensor(data["ALP"], self.T_test)),0)
            mt_bag = torch.cat((mt_bag, self.reshape_tensor(data["MT"], self.T_test)),0)
            sa_bag = torch.cat((sa_bag, self.reshape_tensor(data["SA"], self.T_test)),0)
            ve_bag = torch.cat((ve_bag, self.reshape_tensor(data["VE"], self.T_test)),0)
            rtk_bag = torch.cat((rtk_bag, self.reshape_tensor(data["RTK"], self.T_test)),0)
        
        # We need the initial VE in order to initialize KalmanNet at each batch     
        initial_ve_bag = ve_bag[:,:,0]

        # Shuffle tensors
        if N_T == 0:
            N_T = time_bag.size(0) 
        perm = torch.randperm(time_bag.size(0))[:N_T] # Random permutation

        time_bag = time_bag[perm,:,:]
        imu_bag = imu_bag[perm,:,:]
        ins_bag = ins_bag[perm,:,:]
        gss_bag = gss_bag[perm,:,:]
        rpm_bag = rpm_bag[perm,:,:]
        alp_bag = alp_bag[perm,:,:]
        mt_bag = mt_bag[perm,:,:]
        sa_bag = sa_bag[perm,:,:]
        ve_bag = ve_bag[perm,:,:]
        rtk_bag = rtk_bag[perm,:,:]
        initial_ve_bag = initial_ve_bag[perm,:]

        #Set NaNs to 0
        time_bag[torch.isnan(time_bag)] = 0
        imu_bag[torch.isnan(imu_bag)] = 0
        ins_bag[torch.isnan(ins_bag)] = 0
        gss_bag[torch.isnan(gss_bag)] = 0
        rpm_bag[torch.isnan(rpm_bag)] = 0
        alp_bag[torch.isnan(alp_bag)] = 0
        mt_bag[torch.isnan(mt_bag)] = 0
        sa_bag[torch.isnan(sa_bag)] = 0
        ve_bag[torch.isnan(ve_bag)] = 0
        initial_ve_bag[torch.isnan(initial_ve_bag)] = 0

        self.test_data = {"Time":time_bag, "IMU":imu_bag, "INS":ins_bag, "GSS":gss_bag, "GPD":gpd_bag, "GPS":gps_bag, "RPM":rpm_bag,
                    "ALP":alp_bag, "MT":mt_bag, "SA":sa_bag, "VE":ve_bag, "RTK":rtk_bag, "Initial_VE":initial_ve_bag}

    def reshape_tensor(self,tensor, T):

        # Function to reorganize the long 2nd dimension of the tensors along the 1st dimension
        cut_timestep = tensor.size(1) - tensor.size(1)%T

        split_tensor = torch.split(tensor[:,:cut_timestep], T,dim=1)
        rejoined_tensor = torch.stack(split_tensor, dim=0)

        return rejoined_tensor

    def compute_bias(self,sens,id,calib_time = 400):
        if(id == 'imu'):
            calib_mat = self.C_imu
        elif(id == 'ins'):
            calib_mat = self.C_ins
        else:
            raise('Provide valid ID for bias removal')
        
        # Calibrate sensor
        sens_cal = sens.float()
        sens_cal[[0,1,2],:] = torch.mm(calib_mat, sens_cal[[0,1,2],:])

        # Compute bias for first T samples, we assume that it is 0
        bias = torch.unsqueeze(torch.mean(sens_cal[:,:calib_time],dim=1),dim=1)
        return bias

def rotate_ang(angD,index):

    angR = angD * math.pi/180

    if index == 1:
        C = torch.tensor([[1, 0             , 0              ],
                          [0, math.cos(angR), -math.sin(angR)],
                          [0, math.sin(angR), math.cos(angR) ]])
    elif index == 2:
        C = torch.tensor([[math.cos(angR), 0, math.sin(angR) ],
                          [0,              1,               0],
                          [math.sin(angR), 0, math.cos(angR) ]])
    elif index == 3:
        C = torch.tensor([[math.cos(angR), -math.sin(angR), 0],
                          [math.sin(angR), math.cos(angR) , 0],
                          [0             , 0              , 1]])

    return C

'''
lw = 0.2
lw2 = 0.8
a = torch.cos(torch.unsqueeze(torch.linspace(0, 20, steps=1000),dim=0))
a = torch.unsqueeze(torch.cat([a]*6,dim=0),dim=0)
a = torch.cat([a]*40,dim=0) 
a = a + torch.rand_like(a)
b = gaussian_smoothing(a)
fig, axs = plt.subplots(2, 3, figsize=(30,10))

a = a.detach()
b = b.detach()

axs[0,0].plot(a[1,0,:],'b',linewidth=lw)
axs[0,0].plot(b[1,0,:],'r',linewidth=lw2)

axs[0,1].plot(a[2,1,:],'b',linewidth=lw)
axs[0,1].plot(b[2,1,:],'r',linewidth=lw2)

axs[0,2].plot(a[3,2,:],'b',linewidth=lw)
axs[0,2].plot(b[3,2,:],'r',linewidth=lw2)

axs[1,0].plot(a[0,3,:],'b',linewidth=lw)
axs[1,0].plot(b[0,3,:],'r',linewidth=lw2)

axs[1,1].plot(a[0,4,:],'b',linewidth=lw)
axs[1,1].plot(b[0,4,:],'r',linewidth=lw2)

axs[1,2].plot(a[0,5,:],'b',linewidth=lw)
axs[1,2].plot(b[0,5,:],'r',linewidth=lw2)

plt.show()
'''
