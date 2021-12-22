import torch
import model_parameters as params
import math
from scipy.spatial.transform import Rotation as R
import pymap3d as pm


"""
INPUT: raw observations from all sensors
[ax_imu(0),ay_imu(1),az_imu(2),dyaw_imu(3),ax_ins(4),ay_ins(5),az_ins(6),dyaw_ins(7),rpm_rl(8),rpm_rr(9),rpm_fl(10),rpm_fr(11),4xtm(12,13,14,15),sa(16)]
bool KF_x: if True we need to return the measurements for the x-axis else we have to return the measurements for the y-axis

OUTPUT: [vx, ay]
where vx is approximated through the RPM module
ax is the average of IMU and INS ax reading
"""
def get_sensor_reading(measurement, KF_x, return_yawrate=False):
    ## In this implementation we assume that measurement is the following:
    # [ax_imu(0),ay_imu(1),az_imu(2),dyaw_imu(3),ax_ins(4),ay_ins(5),az_ins(6),dyaw_ins(7),rpm_rl(8),rpm_rr(9),rpm_fl(10),rpm_fr(11),4xtm(12,13,14,15),sa(16)]

    # Step 1: obtain a_x through the IMU and INS
    y_imu = model_imu(measurement[0:4, :])
    y_ins = model_ins(measurement[4:8, :])
    # Step 2: Obtain v_x through the RPM readings
    y_ws = model_rpm(measurement[8:12, :], measurement[16, :])[:, :]

    # Merge all results into 1 vector
    # z_full: [ax_imu, ay_imu, dyaw_imu, ax_ins, ay_ins, dyaw_ins, vy_rpm, vx_rpm, dyaw_rpm]
    z_full = torch.cat((y_imu, y_ins, y_ws), dim=0)

    # Extract v_x
    v_x = z_full[7, :]
    v_y = z_full[6, :]
    # Extract ax_imu and ax_ins and average them
    a_x_imu_1 = z_full[0]
    a_x_imu_2 = z_full[3]
    a_x = 0.5*a_x_imu_1 + 0.5*a_x_imu_2

    ay_imu1 = z_full[1]
    ay_imu2 = z_full[4]
    a_y = 0.5*ay_imu1 + 0.5*ay_imu2

    if(return_yawrate):
        yaw_imu = z_full[2]
        yaw_ins = z_full[5]
        yaw_rpm = z_full[8]
        yaw_average = (yaw_imu+yaw_ins+yaw_rpm)/3
        return yaw_average
    # Save vx, ax as new observation vector
    if(KF_x):
        z = torch.tensor([v_x, a_x])
    else:
        z = torch.tensor([v_y, a_y])
    return z

def model_imu(imu):
    # IMU calibration
    # Input: (ax,ay,az,dyaw)
    imu_calib = torch.zeros_like(imu)
    imu_calib[0:3,:] = torch.mm(params.C_imu, imu[0:3,:])
    imu_calib[3,:] = imu[3,:]
    # Return in format (ax,ay,dyaw)
    return imu_calib[[0,1,3],:]

def model_ins(ins):
    # INS calibration
    # Input: (ax,ay,az,dyaw)
    ins_calib = torch.zeros_like(ins)
    ins_calib[0:3,:] = torch.mm(params.C_ins, ins[0:3,:])
    ins_calib[3,:] = ins[3,:]
    # Return in format (ax,ay,dyaw)
    return ins_calib[[0,1,3],:]


def model_rpm(rpm, new_st):
    ## Steering angle to rads
    new_st = new_st * math.pi / 180

    ## Define inputs
    # Steering of previous timestep is needed for propagation
    # Updated at end of function and stored for next run
    global st
    try:
        st + 0
    except:
        st = new_st
        print("Old_st not working")

    # rpm is 4x1 (RPM of each wheel)
    # Convert to angular velocities
    rps = rpm * (2 * math.pi / 60)
    # Vehicle slow or fast?
    threshold_rpm = 30000

    if all(rpm < threshold_rpm):
        wheelspeed_update = True
    else:
        wheelspeed_update = False

    z_ws = torch.zeros(3, 1)

    if wheelspeed_update:
        # Covert to wheel velocities
        vel = (rps / params.gr_w) * (params.r_tyre * torch.ones(4, 1))[0, 0]
        ang_w = torch.tensor(
            [st / params.gr_s - params.toe[0, 0], st / params.gr_s + params.toe[0, 1], -params.toe[0, 2],
             params.toe[0, 3]])
        ang = torch.mean(ang_w[0:2]).item()
        vx = vel
        vx[0:2, :] = vx[0:2, :] * math.cos(ang)

        # vx
        z_ws[1, 0] = torch.mean(vx)

        yawrate = ((vx[1, 0] - vx[0, 0]) / params.tw + (vx[3, 0] - vx[2, 0]) / params.tw) / 2
        z_ws[0, 0] = yawrate
        # vy
        z_ws[2, 0] = yawrate * params.b

    return z_ws[:, :]

"""
DESCRIPTION: We integrate the velocities vx, vy to obtain the position of the car, we plot this against the GT obtained by MKF
INPUT:
    - x: contains vx and ax for every time step
    - y: contains vy and ay for every time step
    - time:
    - test_input: contains the observations (interested in yaw rate obtained by IMU)
OUTPUT:
    - position in format [[x,y,heading]xT]
"""
def integrate_vel(x, y, time, test_input, T_test):
    # position is an array containing the px,py,heading for each time step
    position = torch.zeros(3, T_test)
    measurement_full = test_input[0, :, :]

    # We start integrating over all time steps
    for i in range(T_test):

        # We define the integration element
        if i == 0:
            dt = time[i + 1] - time[i]
        else:
            dt = time[i] - time[i - 1]

        # We keep track of the yaw_rate, We accumulate the yaw_rate measured by the IMU, multiplied with the integration element
        measurement_i = measurement_full[:, i:i+1] # We take the measurement obtained at time i
        yaw_rate = get_sensor_reading(measurement_i, True, return_yawrate=True)
        dheading = yaw_rate * dt
        if i == 0:
            position[2, i] = dheading
        else:
            position[2, i] = position[2, i-1] + dheading

        # Rotation matrix
        heading = position[2, i]
        r = R.from_euler('z', heading, degrees=False)
        rot_mat = torch.from_numpy(r.as_matrix())[0:2, 0:2].float()

        # Update position
        velocity = torch.tensor([x[0, 0, i:i+1], y[0, 0, i:i+1]]).reshape(2,1) # [v_x, v_y]
        dpos = torch.mm(rot_mat, velocity) * dt
        if i == 0:
            position[0:2, i:i + 1] = dpos
        else:
            position[0:2, i:i + 1] = position[0:2, i - 1:i] + dpos

    return position


def velocity_integration(velocity, time):
    # Input: Velocity in format [[ax,ay,dyaw,vx,vy]xT] and time in [t1,t2,...]
    # velocity = test_target[0, :, :]
    # Output: Position in format [[x,y,heading]xT]
    if torch.cuda.is_available():
        cuda0 = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        cuda0 = torch.device("cpu")

    position = torch.zeros(3, velocity.size(1))

    for i in range(velocity.size(1)):
        if i == 0:
            dt = time[i + 1] - time[i]
        else:
            dt = time[i] - time[i - 1]

        # Update heading
        dheading = velocity[2, i] * dt
        if i == 0:
            position[2, i] = dheading
        else:
            position[2, i] = position[2, i - 1] + dheading

        # Rotation matrix
        heading = position.to(torch.device("cpu"), non_blocking=True)[2, i]
        r = R.from_euler('z', heading, degrees=False)
        rot_mat = torch.from_numpy(r.as_matrix())[0:2, 0:2].float().to(cuda0, non_blocking=True)

        # Update position
        dpos = torch.mm(rot_mat, velocity[3:5, i:i + 1]) * dt
        if i == 0:
            position[0:2, i:i + 1] = dpos
        else:
            position[0:2, i:i + 1] = position[0:2, i - 1:i] + dpos

    return position


def geodetic_transform(geodetic_vec, lat0, lon0):
    # Input: geodetic vec in torch [[lon, lat]xT]
    # Output: enu in [[east, north]xT]
    enu_vec = torch.zeros_like(geodetic_vec)
    for i in range(geodetic_vec.size(1)):
        lat = geodetic_vec[1, i]
        lon = geodetic_vec[0, i]
        h = 0
        h0 = 0
        [enu_vec[0, i], enu_vec[1, i], _] = pm.geodetic2enu(lat, lon, h, lat0, lon0, h0,
                                                            ell=pm.utils.Ellipsoid('wgs84'))

    # Rotation matrix to calibrate with ego motion
    angle = 56.5
    r = R.from_euler('z', angle, degrees=True)
    rot_mat = torch.from_numpy(r.as_matrix())[0:2, 0:2].float()
    enu_rot_vec = torch.mm(rot_mat, enu_vec)

    return enu_rot_vec