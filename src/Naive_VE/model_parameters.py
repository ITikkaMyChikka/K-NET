import math
import torch

import math
import torch

vehicle = True
driver = False

if torch.cuda.is_available():
  cuda0 = torch.device("cuda:0") # you can continue going on here, like cuda:1 cuda:2....etc. 
  torch.set_default_tensor_type('torch.cuda.FloatTensor')
else :
  cuda0 = torch.device("cpu") 

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

#####################################
### Calibration angles in degrees ###
#####################################
gpd_angle = 0.3367
gss_angle = -0.0717
imu_angle = [0.0015,0.0538,-3.3671]
ins_angle = [-0.7187,0.6865,-1.4085]

C_imu = torch.mm(torch.mm(rotate_ang(imu_angle[2], 3),rotate_ang(imu_angle[1], 2)),rotate_ang(imu_angle[0], 1))
C_ins = torch.mm(torch.mm(rotate_ang(ins_angle[2], 3),rotate_ang(ins_angle[1], 2)),rotate_ang(ins_angle[0], 1))
C_gss = rotate_ang(gss_angle*12, 3)[0:1,0:1]
C_gpd = rotate_ang(gpd_angle*0, 3)[0:1,0:1]

####################################
##### Initial sensor positions #####
####################################
if vehicle:
    gss_init_pos = torch.tensor([[0.7888],[0.2475],[0]])
    gpd_init_pos = torch.tensor([[1.1662],[0],[0]])
else:
    gss_init_pos = torch.tensor([[1.0],[0.23],[0]])
    gpd_init_pos = torch.tensor([[0.8],[0],[0]])

imu_init_pos = torch.tensor([[0],[0],[0]])
ins_init_pos = torch.tensor([[0],[0],[0]])
# gps = torch.tensor([0.8947;0;0];
gps_init_pos = torch.tensor([[-0.14],[0],[0]]) # Is actually at the IMU and not somewhere else

#####################################
#### Vehicle Dynamics parameters ####
#####################################
## Values
g = 9.81                    # Gravity (m/s2)
rho = 1.225                 # Density of air (kg/m3)
M = 182 + (200-182)*vehicle + 75*driver # Mass of car (Kg)
r_tyre = (18.3/2)*0.0254    # Static radius of tyre (m)
h = 0.26
h = h                       # COG height (m)
I = 120                     # Yaw inertia of car (kgm2)
Iw = 0.4                    # Rotaional inertia of wheel (kgm2)

wb = 1.530                        
wt_front = 90                      # Weight of front axle (kg)
wt_rear  = 92                      # Weight of rear axle (kg)
a = wt_rear/(wt_front+wt_rear)*wb
b = wb-a
wb = wb                     # Wheelbase (m)
a = a                       # COM distance from front axle (m)
b = b                       # COM distance from rear axle (m)

tw = 1.220
r_l = 0.5*tw
r_r = tw-r_l
tw = tw                     # Trackwidth (m)
r_l = r_l                   # COM distance from left wheels (m)
r_r = r_r                   # COM distance from right wheels (m)

# toe is assumed as constant for all conditions
toe = torch.tensor([[0],[0],[0],[0]]).T*math.pi/180     # Wheel toe angle [FL, FR, RL, RR].T (rad) (+ve: toe-in)

Cd = 1.34 + (1.118-1.34)*vehicle    # Drag Coefficient 
Cl = 3.33 + (3.2-3.33)*vehicle      # Lift Coefficient (down +ve)

A = 1.2                     # Reference Area (m2)

h_a = h                     # COP height (m)
a_a = 0.52*wb
a_a = a_a                   # COP distance from front axle (m)
b_a = wb-a_a                # COP distance from rear axle (m)

# Vectors from COG to wheels
r_w = torch.zeros(3,4)
r_w[:,0] = torch.tensor([+a,+r_l,0])
r_w[:,1] = torch.tensor([+a,-r_r,0])
r_w[:,2] = torch.tensor([-b,+r_l,0])
r_w[:,3] = torch.tensor([-b,-r_r,0])
r_w = r_w

# Gear ratios
gr_w = 14.375 + (18.28-14.375)*vehicle  # Gear ratio from motor to wheel
gr_s = 4                                # Steering angle to delta gear ratio

#####################################
###### Propagation parameters #######
#####################################

Ts = 1/200 # Sampling time in sec
UKF = False
# State.ve: [vx;vy;dyaw;ax;ay;ddyaw]
Q_ve_min = torch.diag(torch.tensor([0.001,0.004,0.04,0.5,0.5]))
# Q_ve_min = blkdiag(0.001,0.004,0.4,0.5,0.5);

adaptive = False
scaling = torch.tensor([[0.05],[0.05],[0.0354],[1],[0.0012532]]).T
# Scaling factors:
# 0.05 = 1/2g - 2g acceleration
# 0.0354 = (1/20)*sqrt(1/0.5)/2 - 20 m/s Vx and 120 deg steering angle
# 1 = -1to+1 dc
# 0.0012532 = 0.0354^2

vmin = 0.5     # Minimum velocity for SR
