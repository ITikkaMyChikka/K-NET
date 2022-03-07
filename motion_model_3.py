import torch
import parameters as params

"""
INPUT:
    - state vector in the form [x, y, theta, v, a, w]
        theta: heading angle/orientation/yaw_rate
        v : linear velocity
        a : acceleration = v_dot
        phi : turn rate/ steering angle = theta_dot
OUTPUT:
    - state vector after state evolution through function f
        [x, y, theta, v, phi]
DESCRIPTION:
    - F function in state evolution model (motion model)
    - EKFNet paper: nonholonomic motion model, Constant Turn Rate and Acceleration

"""


def f(state):
    motion = torch.zeros_like(state)

    motion[0] = (state[4] / (state[5] ** 2)) * (torch.cos(state[2] + state[5] * params.Ts) - torch.cos(state[2])) \
                + ((state[3] + state[4] * params.Ts) * torch.sin(state[2] + state[5] * params.Ts)
                - state[3] * torch.sin(state[2])) / state[5]

    motion[1] = (state[4] / (state[5] ** 2)) * (torch.sin(state[2] + state[5] * params.Ts) - torch.sin(state[2])) \
                - ((state[3] + state[4] * params.Ts) * torch.cos(state[2] + state[5] * params.Ts)
                - state[3] * torch.cos(state[2])) / state[5]

    motion[2] = state[5]*params.Ts
    motion[3] = state[4]*params.Ts

    new_state = state + motion

    # TODO: This return (5,5) but has to only return (5,1) WHy is this happenening?
    return new_state.float()


"""
INPUT:
    - state vector in the form [x, y, theta, v, a, w]
OUTPUT:
    - observations (guessed with state vectors)
    - [theta, v, a]
DESCRIPTION:
    - H function in observation model
    - We only observe the position (GNSS sensor)

"""


def h(state):

    y = torch.tensor([state[2], state[3], state[4]])  # [phi, v, a]

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
    a = state[4]
    w = state[5]
    T = params.Ts
    cos = torch.cos
    sin = torch.sin

    F = torch.tensor([
        [1, 0, (w*(a*T+v)*cos(p+T*w) + a*(sin(p)-sin(p+T*w)) - v*w*cos(p))/(w**2), (sin(p+T*w) - sin(p))/w, (T*w*sin(p+T*w) + cos(p+T*w) - cos(p))/(w**2), (1/(w*w*w))*((T*(w**2)*(a*T+v)-2*a)*cos(p+T*w)-w*(2*a*T+v)*sin(p+T*w)+2*a*cos(p)+v*w*sin(p))],
        [0, 1, (w*(a*T+v)*sin(p+T*w) + a*(cos(p +T*w)-cos(p)) - v*w*sin(p))/(w**2), (cos(p) - cos(p+T*w))/w, -(1/(w**2))*(-sin(p+T*w) + T*w*cos(p+T*w) + sin(p)), (1/(w*w*w))*((T*(w**2)*(a*T+v)-2*a)*sin(p+T*w)-w*(2*a*T+v)*cos(p+T*w)+2*a*sin(p)-v*w*cos(p))],
        [0, 0, 1, 0, 0, T],
        [0, 0, 0, 1, T, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ]).float()

    return F


def getJacobian_H(state):
    H = torch.tensor([
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0]
    ]).float()
    return H


state_dim = 6
obs_dim = 3
Q = torch.eye(state_dim)
R = torch.eye(obs_dim)
