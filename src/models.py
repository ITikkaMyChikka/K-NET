'''
Defining all necessary sensor models
'''

import torch
import numpy as np
import math

import model_parameters as params

## Motion model
def motion(state):
    '''
    motion = torch.tensor([ [0],
                            [0],
                            [0],
                            [state[0,0] + state[2,0]*state[4,0] + state[2,0]*state[1,0]*params.Ts],
                            [state[1,0] - state[2,0]*state[3,0] - state[2,0]*state[0,0]*params.Ts]]) * params.Ts
    '''
    motion = torch.zeros_like(state)
    motion[3,0] = (state[0,0] + state[2,0]*state[4,0] + state[2,0]*state[1,0]*params.Ts) * params.Ts
    motion[4,0] = (state[1,0] - state[2,0]*state[3,0] - state[2,0]*state[0,0]*params.Ts) * params.Ts

    new_state = state + motion
    return new_state.float()

## IMU sensor model
def model_imu(imu):
    # IMU calibration
    # Input: (ax,ay,az,dyaw)
    imu_calib = torch.zeros_like(imu)
    imu_calib[0:3,:] = torch.mm(params.C_imu, imu[0:3,:])
    imu_calib[3,:] = imu[3,:]
    # Return in format (ax,ay,dyaw)
    return imu_calib[[0,1,3],:]

def h_imu(state):
    H = torch.tensor([[1,0,0,0,0],
                      [0,1,0,0,0],
                      [0,0,1,0,0]]).float()
    y_0 = torch.mm(H,state)
    return y_0

## INS sensor model
def model_ins(ins):
    # INS calibration
    # Input: (ax,ay,az,dyaw)
    ins_calib = torch.zeros_like(ins)
    ins_calib[0:3,:] = torch.mm(params.C_ins, ins[0:3,:])
    ins_calib[3,:] = ins[3,:]
    # Return in format (ax,ay,dyaw)
    return ins_calib[[0,1,3],:]

def h_ins(state):
    H = torch.tensor([[1,0,0,0,0],
                      [0,1,0,0,0],
                      [0,0,1,0,0]]).float()
    y_0 = torch.mm(H,state)
    return y_0

## GSS sensor model
def model_gss(gss):
    # GSS calibration
    gss_calib = torch.mm(params.C_gss, gss)
    return gss_calib

def h_gss(state):
    H = torch.tensor([[1,0,-params.gss_init_pos[0,0],0,0],
                      [0,1, params.gss_init_pos[1,0],0,0]]).float()
    y_0 = torch.mm(H,state)
    return y_0

## GPD sensor model
def model_gpd(gpd):
    ## GPD calibration
    calib_gpd = gpd
    calib_gpd[0:1,:] = torch.mm(params.C_gpd, gpd[0:1,:])

## GPS sensor model
def model_gps(gps):
    return gps

## RPM sensor model
def model_rpm(rpm, model):
    # rpm is 4x1 (RPM of each wheel)
    # Convert to angular velocities
    rps = rpm * (2*math.pi/60)
    # Vehicle slow or fast?
    threshold_rpm = 30000

    if all(rpm < threshold_rpm):
        wheelspeed_update = True
    else:
        wheelspeed_update = False

    z_ws = torch.zeros(3,1)

    if wheelspeed_update:
        # Covert to wheel velocities
        vel = (rps/params.gr_w)*model['r_dyn'][0,0]
        ang = torch.mean(model['ang_w'][0:2]).item()
        vx = vel
        vx[0:2,:] = vx[0:2,:]*math.cos(ang)
        
        #vx
        z_ws[1,0] = torch.mean(vx) 

        yawrate = ((vx[1,0]-vx[0,0])/params.tw+(vx[3,0]-vx[2,0])/params.tw)/2
        z_ws[0,0] = yawrate
        #vy
        z_ws[2,0] = yawrate * params.b

    return z_ws[:,:]

def h_rpm(state):
    H = torch.tensor([[0,0,1,0,0],
                      [0,0,0,1,0],
                      [0,0,0,0,1]]).float()
    y_0 = torch.mm(H,state)
    return y_0
    
## ALP sensor model
def model_alp(alp):
    return alp

## MT sensor model
def model_mt(mt):
    return mt

## SA sensor model
def model_sa(sa):
    return sa

## DC sensor model 
def model_dc(dc):
    return dc

## Vehicle dynamics model
def VD_Mean_Model(state,new_st):
    # Vehicle dynamics model of the mean state - all will be too comp heavy
    # Inputs:   mean1 - state.mean consisting of all states
    #           params - vehicle parameters
    #           meas - measurements at current and previous time instant (only for st)
    # Outputs:  model - Vehicle dynamics model for mean state

    ## Load State
    # State:[vx;vy;dyaw;ax;ay;ddyaw;srfl;srfr;srrl;srrr;roll;pitch;droll;dpitch]
    vel =  torch.zeros(3,1)
    vel[0:2,:] = state[3:5,:]
    acc =  torch.zeros((3,1))
    acc[0:2,:] = state[0:2,:]
    drpy =  torch.zeros((3,1))
    drpy[2:3,:] = state[2:3,:]

    ## Steering angle to rads
    new_st = new_st * math.pi/180

    ## Define inputs
    # Steering of previous timestep is needed for propagation
    # Updated at end of function and stored for next run
    global st
    try:
        st + 0
    except:
        st = new_st
        print("Old_st not working")
    

    ## Initialisation
    sa_w = torch.zeros(4,1)
    w_F_w = torch.zeros(3,4)
    b_F_w = torch.zeros(3,4)
    b_V_w = torch.zeros(3,4)
    w_V_w = torch.zeros(3,4)
    Fa = torch.zeros(3,1)
    WTlat = torch.zeros(2,1)
    WTlon = torch.zeros(2,1)
    Cwb = torch.zeros(3,3,4)

    ## Aero Forces
    Fa[0,0] = params.Cd*params.A*(0.5*params.rho*vel[0,:]**2)
    Fa[1,0] = 0 # Assuming no drag in lateral direction
    Fa[2,0] = params.Cl*params.A*(0.5*params.rho*vel[0,:]**2)

    ## Weight Transfer
    WTlat[0,0] = (params.g*params.r_r-acc[1,0]*params.h)/(params.g*params.tw)
    WTlat[1,0] = (params.g*params.r_l+acc[1,0]*params.h)/(params.g*params.tw)
    WTlon[0,0] = (params.g*params.b  -acc[0,0]*params.h +(Fa[2,0]/params.M)*params.b_a-(Fa[0,0]/params.M)*params.h_a)/(params.g*params.wb)
    WTlon[1,0] = (params.g*params.a  +acc[0,0]*params.h +(Fa[2,0]/params.M)*params.a_a+(Fa[0,0]/params.M)*params.h_a)/(params.g*params.wb)
        
    ## Wheel Kinematics
    # Wheel order - 1.FL; 2.FR; 3.RL; 4.RR
    # Wheel heading angle
    ## Calculate angle
    # ang_w = calc_ang(st,params); # Function to convert to wheel angles
    # # Ackermann is steering to left wheel - for right, use negative
    # persistent ackermann
    # if isempty(ackermann)
    #     ackermann = coder.load('Ackermann.csv');
    # end
    # del = zeros(2,1);
    # del(1) = interp1q(+ackermann(:,1),+ackermann(:,2),st);
    # del(2) = interp1q(-ackermann(:,1),-ackermann(:,2),st);

    ang_w = torch.tensor([st/params.gr_s - params.toe[0,0], st/params.gr_s + params.toe[0,1],-params.toe[0,2], params.toe[0,3]])

    ## Slip angle and Slip Ratio
    r_dyn = params.r_tyre * torch.ones(4,1) # Replace with calibration from free rolling wheel if possible

    for i in range(4):
        
        # Indices to access each tyre
        ind1 = 1-math.ceil(i/2)%2; 
        ind2 = 1-i%2
        
        # Velocities at each tyre in body frame
        b_V_w[:,i:i+1] = vel + torch.tensor([[-drpy[2,0]*params.r_w[1,i]],[drpy[2,0]*params.r_w[0,i]],[0]])
        
        # Rotation matrices for each tyre
        Cwb[:,:,i] = torch.tensor([[math.cos(ang_w[i]), math.sin(ang_w[i]), 0],[-math.sin(ang_w[i]), math.cos(ang_w[i]), 0],[0,0,1]])
        
        # Velocities at each tyre in wheel frame
        w_V_w[:,i:i+1] = torch.mm(Cwb[:,:,i],b_V_w[:,i:i+1])
        
        # Sa at each wheel
        sa_w[i,0] = math.atan2(b_V_w[1,i],b_V_w[0,i]) - ang_w[i]
        
        # Wheel forces in wheel frame
        w_F_w[2,i] = params.M*params.g * WTlat[ind1,0] * WTlon[ind2,0]

    ## Wheel Forces and Tyre Model
    # Fx and Fy of each tyre obtained from Tyre Model
    # Input - Fz on each tyre(w_F(3,:)), SA, SR 
    # Output - Fx and Fy of each tyre (w_F(1:2,:))

    # for i = 1:4
    #     [w_F_w(1,i),w_F_w(2,i)] = tyre_forces(sr_w(i),sa_w(i),w_F_w(3,i));
    # end

    # Convert to body frame

    for i in range(4):
        b_F_w[:,i] = torch.matmul(Cwb[:,:,i].T, w_F_w[:,i])

    ## Outputs
    model = {"ang_w": ang_w, "Cwb": Cwb, "Fa":Fa, "b_V_w": b_V_w, "w_V_w": w_V_w, "b_F_w": b_F_w, "w_F_w": w_F_w, "r_dyn": r_dyn}

    # Save steering for next function call
    st = new_st

    return model

