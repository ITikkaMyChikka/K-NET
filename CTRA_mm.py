import torch
import src.parameters as params

T = params.T_ctra
T_dec = T
#T_dec = T*params.ratio_ctra
q = params.q_ctra

sin = torch.sin
cos = torch.cos
# state = [x, y, phi, s, a, w]

sa = params.sa
sw = params.sw


#######################
### DATA GENERATION ###
#######################

def F_gen(state):
    x = state[0]  # pos x
    y = state[1]  # pos y
    p = state[2]  # phi
    s = state[3]  # speed
    a = state[4]  # longitudinal acceleration
    w = state[5]  # yaw rate

    motion = torch.zeros_like(state)

    motion[0] = (1/w)*(s+a*T)*sin(p+w*T) + (a/w**2)*cos(p+w*T) - (s/w)*sin(p) - (a/w**2)*cos(p)
    motion[1] = (1/w)*(-s-a*T)*cos(p+w*T) + (a/w**2)*sin(p+w*T) + (s/w)*cos(p) - (a/w**2)*sin(p)
    motion[2] = w*T
    motion[3] = a*T

    new_state = state + motion
    return new_state.float()

def Q_gen(state):
    Q_gen = torch.tensor([[0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0], ]).float()
    return Q_gen



def H_gen(state):
    H_gen = torch.tensor([[1, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1]]).float()
    return H_gen

def R_gen(state):

    R_gen = torch.tensor([[0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]]).float()
    return R_gen



##############################
### FULL OBSERVATION MODEL ###
##############################

def F_FO(state):
    x = state[0]  # pos x
    y = state[1]  # pos y
    p = state[2]  # phi
    s = state[3]  # speed
    a = state[4]  # longitudinal acceleration
    w = state[5]  # yaw rate

    motion = torch.zeros_like(state)

    motion[0] = (1 / w) * (s + a * T_dec) * sin(p + w * T_dec) + (a / w ** 2) * cos(p + w * T_dec) - (s / w) * sin(p) - (
                a / w ** 2) * cos(p)
    motion[1] = (1 / w) * (-s - a * T_dec) * cos(p + w * T_dec) + (a / w ** 2) * sin(p + w * T_dec) + (s / w) * cos(p) - (
                a / w ** 2) * sin(p)
    motion[2] = w * T_dec
    motion[3] = a * T_dec

    new_state = state + motion
    return new_state.float()

def Q_FO(state):
    x = state[0]  # pos x
    y = state[1]  # pos y
    p = state[2]  # phi
    s = state[3]  # speed
    a = state[4]  # longitudinal acceleration
    w = state[5]  # yaw rate

    x1 = (T_dec**5/20)*(sw**2*s**2*(sin(p)**2) + sa**2*(cos(p))**2)  # xx
    x2 = (T_dec**5/20)*sin(p)*cos(p)*(sa**2 - sw**2*s**2)   # xy
    x3 = -(T_dec**4/8)*sw**2*s*sin(p) # xphi
    x4 = (T_dec**4/8)*sa**2*cos(p)  # xs
    x5 =  (T_dec**3/6)*sa**2*cos(p)  # xa
    x6 =  -(T_dec**3/6)*sw**2*s*sin(p)  # xw

    y1 = (T_dec**5/20)*sin(p)*cos(p)*(sa**2 - sw**2*s**2)
    y2 = (T_dec**5/20)*(sw**2*s**2*(cos(p)**2) + sa**2*(sin(p))**2)
    y3 = (T_dec**4/8)*sw**2*s*cos(p)
    y4 = (T_dec**4/8)*sa**2*sin(p)
    y5 = (T_dec**3/6)*sa**2*sin(p)
    y6 = (T_dec**3/6)*sw**2*s*cos(p)

    p1 = -(T_dec**4/8)*sw**2*s*sin(p)
    p2 = (T_dec**4/8)*sw**2*s*cos(p)
    p3 = (T_dec**3/3)*sw**2
    p4 = 0
    p5 = 0
    p6 = (T_dec**2/2)*sw**2

    s1 = (T_dec**4/8)*sa**2*cos(p)
    s2 = (T_dec**4/8)*sa**2*sin(p)
    s3 = 0
    s4 = (T_dec**3/3)*sa**2
    s5 = (T_dec**2/2)*sa**2
    s6 = 0

    a1 = (T_dec**3/6)*sa**2*cos(p)
    a2 = (T_dec**3/6)*sa**2*sin(p)
    a3 = 0
    a4 = (T_dec**2/2)*sa**2
    a5 = T_dec**sa
    a6 = 0

    w1 = -(T_dec**3/6)*sw**2*s*sin(p)
    w2 = (T_dec**3/6)*sw**2*s*cos(p)
    w3 = (T_dec**2/2)*sw**2
    w4 = 0
    w5 = 0
    w6 = T_dec*sw**2

    Q_FO = torch.tensor([[x1, x2, x3, x4, x5, x6],
                         [y1, y2, y3, y4, y5, y6],
                         [p1, p2, p3, p4, p5, p6],
                         [s1, s2, s3, s4, s5, s6],
                         [a1, a2, a3, a4, a5, a6],
                         [w1, w2, w3, w4, w5, w6]]).float()
    return Q_FO

def H_FO(state):

    H_FO = torch.tensor([ [1, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1]]).float()
    return H_FO

def R_FO(state):

    R_FO = torch.tensor([[1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1]]).float()
    return R_FO




############################
### POSITION OBSERVATION ###
############################

def F_PO(state):
    x = state[0]  # pos x
    y = state[1]  # pos y
    p = state[2]  # phi
    s = state[3]  # speed
    a = state[4]  # longitudinal acceleration
    w = state[5]  # yaw rate

    motion = torch.zeros_like(state)

    motion[0] = (1 / w) * (s + a * T_dec) * sin(p + w * T_dec) + (a / w ** 2) * cos(p + w * T_dec) - (s / w) * sin(p) - (
                a / w ** 2) * cos(p)
    motion[1] = (1 / w) * (-s - a * T_dec) * cos(p + w * T_dec) + (a / w ** 2) * sin(p + w * T_dec) + (s / w) * cos(p) - (
                a / w ** 2) * sin(p)
    motion[2] = w * T_dec
    motion[3] = a * T_dec

    new_state = state + motion
    return new_state.float()

def Q_PO(state):
    x = state[0]  # pos x
    y = state[1]  # pos y
    p = state[2]  # phi
    s = state[3]  # speed
    a = state[4]  # longitudinal acceleration
    w = state[5]  # yaw rate

    x1 = (T_dec**5/20)*(sw**2*s**2*(sin(p)**2) + sa**2*(cos(p))**2)  # xx
    x2 = (T_dec**5/20)*sin(p)*cos(p)*(sa**2 - sw**2*s**2)   # xy
    x3 = -(T_dec**4/8)*sw**2*s*sin(p) # xphi
    x4 = (T_dec**4/8)*sa**2*cos(p)  # xs
    x5 =  (T_dec**3/6)*sa**2*cos(p)  # xa
    x6 =  -(T_dec**3/6)*sw**2*s*sin(p)  # xw

    y1 = (T_dec**5/20)*sin(p)*cos(p)*(sa**2 - sw**2*s**2)
    y2 = (T_dec**5/20)*(sw**2*s**2*(cos(p)**2) + sa**2*(sin(p))**2)
    y3 = (T_dec**4/8)*sw**2*s*cos(p)
    y4 = (T_dec**4/8)*sa**2*sin(p)
    y5 = (T_dec**3/6)*sa**2*sin(p)
    y6 = (T_dec**3/6)*sw**2*s*cos(p)

    p1 = -(T_dec**4/8)*sw**2*s*sin(p)
    p2 = (T_dec**4/8)*sw**2*s*cos(p)
    p3 = (T_dec**3/3)*sw**2
    p4 = 0
    p5 = 0
    p6 = (T_dec**2/2)*sw**2

    s1 = (T_dec**4/8)*sa**2*cos(p)
    s2 = (T_dec**4/8)*sa**2*sin(p)
    s3 = 0
    s4 = (T_dec**3/3)*sa**2
    s5 = (T_dec**2/2)*sa**2
    s6 = 0

    a1 = (T_dec**3/6)*sa**2*cos(p)
    a2 = (T_dec**3/6)*sa**2*sin(p)
    a3 = 0
    a4 = (T_dec**2/2)*sa**2
    a5 = T_dec**sa
    a6 = 0

    w1 = -(T_dec**3/6)*sw**2*s*sin(p)
    w2 = (T_dec**3/6)*sw**2*s*cos(p)
    w3 = (T_dec**2/2)*sw**2
    w4 = 0
    w5 = 0
    w6 = T_dec*sw**2

    Q_PO = torch.tensor([[x1, x2, x3, x4, x5, x6],
                         [y1, y2, y3, y4, y5, y6],
                         [p1, p2, p3, p4, p5, p6],
                         [s1, s2, s3, s4, s5, s6],
                         [a1, a2, a3, a4, a5, a6],
                         [w1, w2, w3, w4, w5, w6]]).float()
    return Q_PO

def H_PO(state):

    H_PO = torch.tensor([[1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0]]).float()
    return H_PO

def R_PO(state):
    R_PO = torch.tensor([[1, 0],
                         [0, 1]]).float()
    return R_PO


################################
### ACCELERATION OBSERVATION ###
################################

def F_AO(state):
    x = state[0]  # pos x
    y = state[1]  # pos y
    p = state[2]  # phi
    s = state[3]  # speed
    a = state[4]  # longitudinal acceleration
    w = state[5]  # yaw rate

    motion = torch.zeros_like(state)

    motion[0] = (1 / w) * (s + a * T_dec) * sin(p + w * T_dec) + (a / w ** 2) * cos(p + w * T_dec) - (s / w) * sin(p) - (
                a / w ** 2) * cos(p)
    motion[1] = (1 / w) * (-s - a * T_dec) * cos(p + w * T_dec) + (a / w ** 2) * sin(p + w * T_dec) + (s / w) * cos(p) - (
                a / w ** 2) * sin(p)
    motion[2] = w * T_dec
    motion[3] = a * T_dec

    new_state = state + motion
    return new_state.float()

def Q_AO(state):
    x = state[0]  # pos x
    y = state[1]  # pos y
    p = state[2]  # phi
    s = state[3]  # speed
    a = state[4]  # longitudinal acceleration
    w = state[5]  # yaw rate

    x1 = (T_dec**5/20)*(sw**2*s**2*(sin(p)**2) + sa**2*(cos(p))**2)  # xx
    x2 = (T_dec**5/20)*sin(p)*cos(p)*(sa**2 - sw**2*s**2)   # xy
    x3 = -(T_dec**4/8)*sw**2*s*sin(p) # xphi
    x4 = (T_dec**4/8)*sa**2*cos(p)  # xs
    x5 =  (T_dec**3/6)*sa**2*cos(p)  # xa
    x6 =  -(T_dec**3/6)*sw**2*s*sin(p)  # xw

    y1 = (T_dec**5/20)*sin(p)*cos(p)*(sa**2 - sw**2*s**2)
    y2 = (T_dec**5/20)*(sw**2*s**2*(cos(p)**2) + sa**2*(sin(p))**2)
    y3 = (T_dec**4/8)*sw**2*s*cos(p)
    y4 = (T_dec**4/8)*sa**2*sin(p)
    y5 = (T_dec**3/6)*sa**2*sin(p)
    y6 = (T_dec**3/6)*sw**2*s*cos(p)

    p1 = -(T_dec**4/8)*sw**2*s*sin(p)
    p2 = (T_dec**4/8)*sw**2*s*cos(p)
    p3 = (T_dec**3/3)*sw**2
    p4 = 0
    p5 = 0
    p6 = (T_dec**2/2)*sw**2

    s1 = (T_dec**4/8)*sa**2*cos(p)
    s2 = (T_dec**4/8)*sa**2*sin(p)
    s3 = 0
    s4 = (T_dec**3/3)*sa**2
    s5 = (T_dec**2/2)*sa**2
    s6 = 0

    a1 = (T_dec**3/6)*sa**2*cos(p)
    a2 = (T_dec**3/6)*sa**2*sin(p)
    a3 = 0
    a4 = (T_dec**2/2)*sa**2
    a5 = T_dec**sa
    a6 = 0

    w1 = -(T_dec**3/6)*sw**2*s*sin(p)
    w2 = (T_dec**3/6)*sw**2*s*cos(p)
    w3 = (T_dec**2/2)*sw**2
    w4 = 0
    w5 = 0
    w6 = T_dec*sw**2

    Q_PO = torch.tensor([[x1, x2, x3, x4, x5, x6],
                         [y1, y2, y3, y4, y5, y6],
                         [p1, p2, p3, p4, p5, p6],
                         [s1, s2, s3, s4, s5, s6],
                         [a1, a2, a3, a4, a5, a6],
                         [w1, w2, w3, w4, w5, w6]]).float()
    return Q_PO

def H_AO(state):
    H_AO = torch.tensor([[0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1]]).float()
    return H_AO

def R_AO(state):
    R_AO = torch.tensor([[1, 0],
                     [0, 1]]).float()
    return R_AO


#############################################
### POSITION AND ACCELERATION OBSERVATION ###
#############################################
def F_PAO(state):
    x = state[0]  # pos x
    y = state[1]  # pos y
    p = state[2]  # phi
    s = state[3]  # speed
    a = state[4]  # longitudinal acceleration
    w = state[5]  # yaw rate

    motion = torch.zeros_like(state)

    motion[0] = (1 / w) * (s + a * T_dec) * sin(p + w * T_dec) + (a / w ** 2) * cos(p + w * T_dec) - (s / w) * sin(p) - (
                a / w ** 2) * cos(p)
    motion[1] = (1 / w) * (-s - a * T_dec) * cos(p + w * T_dec) + (a / w ** 2) * sin(p + w * T_dec) + (s / w) * cos(p) - (
                a / w ** 2) * sin(p)
    motion[2] = w * T_dec
    motion[3] = a * T_dec

    new_state = state + motion
    return new_state.float()

def Q_PAO(state):
    x = state[0]  # pos x
    y = state[1]  # pos y
    p = state[2]  # phi
    s = state[3]  # speed
    a = state[4]  # longitudinal acceleration
    w = state[5]  # yaw rate

    x1 = (T_dec**5/20)*(sw**2*s**2*(sin(p)**2) + sa**2*(cos(p))**2)  # xx
    x2 = (T_dec**5/20)*sin(p)*cos(p)*(sa**2 - sw**2*s**2)   # xy
    x3 = -(T_dec**4/8)*sw**2*s*sin(p) # xphi
    x4 = (T_dec**4/8)*sa**2*cos(p)  # xs
    x5 =  (T_dec**3/6)*sa**2*cos(p)  # xa
    x6 =  -(T_dec**3/6)*sw**2*s*sin(p)  # xw

    y1 = (T_dec**5/20)*sin(p)*cos(p)*(sa**2 - sw**2*s**2)
    y2 = (T_dec**5/20)*(sw**2*s**2*(cos(p)**2) + sa**2*(sin(p))**2)
    y3 = (T_dec**4/8)*sw**2*s*cos(p)
    y4 = (T_dec**4/8)*sa**2*sin(p)
    y5 = (T_dec**3/6)*sa**2*sin(p)
    y6 = (T_dec**3/6)*sw**2*s*cos(p)

    p1 = -(T_dec**4/8)*sw**2*s*sin(p)
    p2 = (T_dec**4/8)*sw**2*s*cos(p)
    p3 = (T_dec**3/3)*sw**2
    p4 = 0
    p5 = 0
    p6 = (T_dec**2/2)*sw**2

    s1 = (T_dec**4/8)*sa**2*cos(p)
    s2 = (T_dec**4/8)*sa**2*sin(p)
    s3 = 0
    s4 = (T_dec**3/3)*sa**2
    s5 = (T_dec**2/2)*sa**2
    s6 = 0

    a1 = (T_dec**3/6)*sa**2*cos(p)
    a2 = (T_dec**3/6)*sa**2*sin(p)
    a3 = 0
    a4 = (T_dec**2/2)*sa**2
    a5 = T_dec**sa
    a6 = 0

    w1 = -(T_dec**3/6)*sw**2*s*sin(p)
    w2 = (T_dec**3/6)*sw**2*s*cos(p)
    w3 = (T_dec**2/2)*sw**2
    w4 = 0
    w5 = 0
    w6 = T_dec*sw**2

    Q_PO = torch.tensor([[x1, x2, x3, x4, x5, x6],
                         [y1, y2, y3, y4, y5, y6],
                         [p1, p2, p3, p4, p5, p6],
                         [s1, s2, s3, s4, s5, s6],
                         [a1, a2, a3, a4, a5, a6],
                         [w1, w2, w3, w4, w5, w6]]).float()
    return Q_PO

def H_PAO(state):
    H_PAO = torch.tensor([[1, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1]]).float()
    return H_PAO

def R_PAO(state):

    R_PAO = torch.tensor([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]]).float()
    return R_PAO



r_FO = torch.tensor([[1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1]]).float()


################
### JACOBIAN ###
################

def F_jacobian(state):
    x = state[0]  # pos x
    y = state[1]  # pos y
    p = state[2]  # phi
    s = state[3]  # speed
    a = state[4]  # longitudinal acceleration
    w = state[5]  # yaw rate

    f1x = 1
    f1y = 0
    f1p = (1/w)*(a*T_dec+s)*cos(p+T_dec*w) + (1/w**2)*a*(sin(p)-sin(p+T_dec*w)) - (1/w)*cos(p)*s
    f1s = (1/w) * (sin(p+T_dec*w) - sin(p))
    f1a = (1/w) * (T_dec*sin(p+T_dec*w)) + (1/w**2)*(cos(p+T_dec*w)-cos(p))
    f1w = (1/w)* (T_dec*(T_dec*a + s)*cos(p+T_dec*w)) + (1/w**2)*(s*sin(p) - a*T_dec*sin(p+T_dec*w) - (a*T_dec+a)*sin(p+T_dec*w))

    f2x = 0
    f2y = 1
    f2p = (1/w)*((a*T_dec+s)*sin(p+T_dec*w) - sin(p)*s) + (1/w**2)*(a*(cos(p+T_dec*w)-cos(p)))
    f2s = (1/w)*( -cos(p+T_dec*w) + cos(p))
    f2a = (1/w)*(-T_dec*cos(p+T_dec*w)) + (1/w**2)*(sin(p+T_dec*w)-sin(p))
    f2w = (1/w)*T_dec*sin(p+w*T_dec)*(s+a*T_dec) + (1/w**2)*cos(p+T_dec*w)*(2*a*T_dec+s) -(1/w**2)*(s*cos(p)) + (1/w**3)*2*a*(sin(p)-sin(p+T_dec*w))

    f3x = 0
    f3y = 0
    f3p = 1
    f3s = 0
    f3a = 0
    f3w = T_dec

    f4x = 0
    f4y = 0
    f4p = 0
    f4s = 1
    f4a = T_dec
    f4w = 0

    f5x = 0
    f5y = 0
    f5p = 0
    f5s = 0
    f5a = 1
    f5w = 0

    f6x = 0
    f6y = 0
    f6p = 0
    f6s = 0
    f6a = 0
    f6w = 1

    F = torch.tensor([[f1x, f1y, f1p, f1s, f1a, f1w],
                         [f2x, f2y, f2p, f2s, f2a, f2w],
                         [f3x, f3y, f3p, f3s, f3a, f3w],
                         [f4x, f4y, f4p, f4s, f4a, f4w],
                         [f5x, f5y, f5p, f5s, f5a, f5w],
                         [f6x, f6y, f6p, f6s, f6a, f6w]]).float()
    return F

def F_jacobian_smooth(state):

    x = state[0]  # pos x
    y = state[1]  # pos y
    p = state[2]  # phi
    s = state[3]  # speed
    a = state[4]  # longitudinal acceleration
    w = state[5]  # yaw rate

    f1x = 1
    f1y = 0
    f1p = (1/w)*(a*T+s)*cos(p+T*w) + (1/w**2)*a*(sin(p)-sin(p+T*w)) - (1/w)*cos(p)*s
    f1s = (1/w) * (sin(p+T*w) - sin(p))
    f1a = (1/w) * (T*sin(p+T*w)) + (1/w**2)*(cos(p+T*w)-cos(p))
    f1w = (1/w)* (T*(T*a + s)*cos(p+T*w)) + (1/w**2)*(s*sin(p) - a*T*sin(p+T*w) - (a*T+a)*sin(p+T*w))

    f2x = 0
    f2y = 1
    f2p = (1/w)*((a*T+s)*sin(p+T*w) - sin(p)*s) + (1/w**2)*(a*(cos(p+T*w)-cos(p)))
    f2s = (1/w)*( -cos(p+T*w) + cos(p))
    f2a = (1/w)*(-T*cos(p+T*w)) + (1/w**2)*(sin(p+T*w)-sin(p))
    f2w = (1/w)*T*sin(p+w*T)*(s+a*T) + (1/w**2)*cos(p+T*w)*(2*a*T+s) -(1/w**2)*(s*cos(p)) + (1/w**3)*2*a*(sin(p)-sin(p+T*w))

    f3x = 0
    f3y = 0
    f3p = 1
    f3s = 0
    f3a = 0
    f3w = T

    f4x = 0
    f4y = 0
    f4p = 0
    f4s = 1
    f4a = T
    f4w = 0

    f5x = 0
    f5y = 0
    f5p = 0
    f5s = 0
    f5a = 1
    f5w = 0

    f6x = 0
    f6y = 0
    f6p = 0
    f6s = 0
    f6a = 0
    f6w = 1

    F = torch.tensor([[f1x, f1y, f1p, f1s, f1a, f1w],
                         [f2x, f2y, f2p, f2s, f2a, f2w],
                         [f3x, f3y, f3p, f3s, f3a, f3w],
                         [f4x, f4y, f4p, f4s, f4a, f4w],
                         [f5x, f5y, f5p, f5s, f5a, f5w],
                         [f6x, f6y, f6p, f6s, f6a, f6w]]).float()
    return F


sGPS     = 0.5*8.8*T**2  # assume 8.8m/s2 as maximum acceleration, forcing the vehicle
sCourse  = 0.1*T # assume 0.1rad/s as maximum turn rate for the vehicle
sVelocity= 8.8*T # assume 8.8m/s2 as maximum acceleration, forcing the vehicle
sYaw     = 1.0*T # assume 1.0rad/s2 as the maximum turn rate acceleration for the vehicle
sAccel   = 0.5

q_gen = torch.diag(torch.tensor([sGPS**2, sGPS**2, sCourse**2, sVelocity**2, sAccel**2,  sYaw**2 ]))


sGPS     = 0.5*8.8*T_dec**2  # assume 8.8m/s2 as maximum acceleration, forcing the vehicle
sCourse  = 0.1*T_dec # assume 0.1rad/s as maximum turn rate for the vehicle
sVelocity= 8.8*T_dec # assume 8.8m/s2 as maximum acceleration, forcing the vehicle
sYaw     = 1.0*T_dec # assume 1.0rad/s2 as the maximum turn rate acceleration for the vehicle
sAccel   = 0.5

q_simple = torch.diag(torch.tensor([sGPS**2, sGPS**2, sCourse**2, sVelocity**2, sAccel**2,  sYaw**2]))



