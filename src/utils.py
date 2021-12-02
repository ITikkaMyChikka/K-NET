import pymap3d as pm
import torch
from scipy.spatial.transform import Rotation as R

def remove_bias(imu, ins, flag):
    # Function to calculate biases of accelerometers and gyros
    # Input:    imu - IMU measurements
    #           ins - INS measurements
    #           flag - flag to change biases
    # Output:   bias - bias structure containing all sensors

    ##
    global count, count_imu, count_ins, bias_imu, bias_ins, old_imu, old_ins
    if count is None:
        count = 0
        count_imu = 0
        count_ins = 0
        bias_imu  = torch.zeros(6,1)
        bias_ins  = torch.zeros(6,1)
        old_imu   = torch.zeros(6,1)
        old_ins   = torch.zeros(6,1)

    if any(imu is not old_imu) and flag and count>500 and count<900:
        bias_imu = (bias_imu*count_imu + imu)/(count_imu+1)
        count_imu = count_imu + 1


    if any(ins is not old_ins) and flag and count>500 and count<900:
        bias_ins = (bias_ins*count_ins + ins)/(count_ins+1)
        count_ins = count_ins + 1

    corr_imu = imu - bias_imu
    corr_ins = ins - bias_ins
    old_imu = imu
    old_ins = ins
    if flag and count<1000:
        count = count+1

    count_out = count

    return corr_imu, corr_ins, count_out

def velocity_integration(velocity, time):
    # Input: Velocity in format [[ax,ay,dyaw,vx,vy]xT] and time in [t1,t2,...]
    # Output: Position in format [[x,y,heading]xT]
    if torch.cuda.is_available():
      cuda0 = torch.device("cuda:0") # you can continue going on here, like cuda:1 cuda:2....etc. 
      torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else :
      cuda0 = torch.device("cpu") 

    position = torch.zeros(3,velocity.size(1))

    for i in range(velocity.size(1)):
        if i == 0:
            dt = time[i+1] - time[i]
        else:
            dt = time[i] - time[i-1]

        # Update heading
        dheading = velocity[2,i] * dt
        if i == 0:
            position[2,i] = dheading
        else:
            position[2,i]  = position[2,i-1] + dheading

        # Rotation matrix
        heading = position.to(torch.device("cpu"),non_blocking=True)[2,i]
        r = R.from_euler('z', heading, degrees=False)
        rot_mat = torch.from_numpy(r.as_matrix())[0:2,0:2].float().to(cuda0,non_blocking=True)

        # Update position
        dpos = torch.mm(rot_mat,velocity[3:5,i:i+1]) * dt
        if i == 0:
            position[0:2,i:i+1] = dpos
        else:
            position[0:2,i:i+1] = position[0:2,i-1:i] + dpos
    
    return position

def geodetic_transform(geodetic_vec, lat0, lon0):
    # Input: geodetic vec in torch [[lon, lat]xT]
    # Output: enu in [[east, north]xT]
    enu_vec = torch.zeros_like(geodetic_vec)
    for i in range(geodetic_vec.size(1)):
        lat = geodetic_vec[1,i]
        lon = geodetic_vec[0,i]
        h = 0
        h0 = 0
        [enu_vec[0,i],enu_vec[1,i],_] = pm.geodetic2enu(lat, lon, h, lat0, lon0, h0, ell=pm.utils.Ellipsoid('wgs84'))
    
    # Rotation matrix to calibrate with ego motion
    angle = 56.5
    r = R.from_euler('z', angle, degrees=True)
    rot_mat = torch.from_numpy(r.as_matrix())[0:2,0:2].float()
    enu_rot_vec = torch.mm(rot_mat,enu_vec)

    return enu_rot_vec

