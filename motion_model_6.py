import torch
import parameters as params

"""
INPUT:
    - state vector in the form [x, y, theta, v, w]
        theta: heading angle/orientation/yaw_rate
        v : linear velocity
        phi : turn rate/ steering angle = theta_dot
OUTPUT:
    - state vector after state evolution through function f
        [x, y, theta, v, w]
DESCRIPTION:
    - F function in state evolution model (motion model)
    - EKFNet paper: nonholonomic motion model, Constant Turn Rate and Velocity

"""


def f(state):
    motion = torch.zeros_like(state)
    p = state[2]
    v = state[3]
    w = state[4]

    motion[0] = (v/w)*(torch.sin(p+w*params.Ts)-torch.sin(p))
    motion[1] = (v/w)*(torch.cos(p)-torch.cos(p+w*params.Ts))
    motion[2] = w*params.Ts

    new_state = state + motion

    return new_state.float()


"""
INPUT:
    - state vector in the form [x, y, theta, v, w]
OUTPUT:
    - observations (guessed with state vectors)
    - [x, y, theta, v]
DESCRIPTION:
    - H function in observation model
    - We observe position, velocity, yaw_rate

"""


def h(state):

    y = torch.tensor([state[0], state[1], state[2], state[3]])  # [x, y, theta, v]

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
    p = state[2]
    v = state[3]
    w = state[4]
    T = params.Ts
    cos = torch.cos
    sin = torch.sin

    F = torch.tensor([
        [1, 0, (v/w)*(-cos(p) + cos(p + T*w)) ,(1/w)*(-sin(p) + sin(p + T*w)), (v/w**2)*(-sin(p+T*w) + T*w*cos(p+T*w) + sin(p))],
        [0, 1, (v/w)*(-sin(p) + sin(p + T*w)), (1/w)*(cos(p) - cos(p + T*w)), (v/w**2)*(-cos(p) + cos(p + T*w) + T*w*sin(p+T*w)) ],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]
    ]).float()

    return F


def getJacobian_H(state):
    H = torch.tensor([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
    ]).float()
    return H


state_dim = 5
obs_dim = 4
Q = torch.eye(state_dim)
R = torch.eye(obs_dim)
