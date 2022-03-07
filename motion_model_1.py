import torch
import parameters as params

"""
INPUT:
    - state vector in the form [ax, ay, phi, vx, vy]
OUTPUT:
    - state vector after state evolution through function f
        [ax, ay, phi, vx, vy]
DESCRIPTION:
    - F function in state evolution model (motion model)
    - Adria's State Space Model

"""
def f(state):

    motion = torch.zeros_like(state)

    # Easy model
    """
    motion[3] = (params.Ts * state[2] * state[4] + params.Ts * state[0])  # phi*Ts*vy + Ts*ax
    motion[4] = (-params.Ts * state[2] * state[3] + params.Ts * state[1])  # -phi*Ts*vx + Ts*ay
    """

    motion[3] = state[2]*state[4]*params.Ts + state[0]*params.Ts + state[2]*state[1]*params.Ts*params.Ts
    motion[4] =-state[2]*state[3]*params.Ts + state[1]*params.Ts - state[2]*state[0]*params.Ts*params.Ts
    new_state = state + motion

    # TODO: This return (5,5) but has to only return (5,1) WHy is this happenening?
    return new_state.float()

"""
INPUT:
    - state vector in the form [ax, ay, phi, vx, vy]
OUTPUT:
    - observations (guessed with state vectors)
    - [ax_imu, ay_imu, dyaw_imu, ax_ins, ay_ins, dyaw_ins, vx_rpm, vy_rpm]
DESCRIPTION:
    - H function in observation model
    - Adria's State Space Model

"""
def h(state):
    # First we get the IMU readings
    y_imu = h_imu(state)  # [ax, ay, phi]
    # Second we get the second IMU/ INS readings
    y_ins = h_imu(state)  # [ax, ay, phi]
    # Finally we get the RPM model readings [vx, vy]
    y_rpm = h_rpm(state)

    # We concatenate all the observations
    y = torch.cat((y_imu, y_ins, y_rpm), dim=0)

    return y


def h_rpm(state):
    H = torch.tensor([[0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 1]]).float()
    y_0 = torch.matmul(H, state)
    return y_0


def h_imu(state):
    H = torch.tensor([[1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0],
                      [0, 0, 1, 0, 0]]).float()
    y_0 = torch.matmul(H, state)
    return y_0

def getJacobian_F(state):

    F = torch.tensor([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [params.Ts, state[2]*params.Ts*params.Ts, state[4]*params.Ts+state[1]*params.Ts*params.Ts, 1, state[2]*params.Ts],
        [state[2]*params.Ts*params.Ts, params.Ts, -state[3]*params.Ts-state[0]*params.Ts*params.Ts, 1, -state[2]*params.Ts]
    ]).float()
    return F

def getJacobian_H(state):

    H = torch.tensor([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],

    ]).float()
    return H

state_dim = 5
obs_dim = 8
Q = torch.eye(state_dim)
R = torch.eye(obs_dim)
