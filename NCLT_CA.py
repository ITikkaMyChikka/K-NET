import torch
import src.parameters as params

#T = params.T_cv
#T_dec = T*params.ratio_cv
#q = params.q_cv



T = 0.0001
T_dec = T


#######################
### DATA GENERATION ###
#######################
F = torch.tensor([[1, T, 0.5*T**2, 0, 0, 0],
                          [0, 1, T, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 1, T, 0.5*T**2],
                          [0, 0, 0, 0, 1, T],
                          [0, 0, 0, 0, 0, 1]]).float()

Q = torch.tensor([[T_dec ** 4 / 4, T_dec ** 3 / 2, T_dec ** 2 / 2, 0, 0, 0],
                         [T_dec ** 3 / 2, T_dec ** 2, T_dec, 0, 0, 0],
                         [T_dec ** 2 / 2, T_dec, 1, 0, 0, 0],
                         [0, 0, 0, T_dec ** 4 / 4, T_dec ** 3 / 2, T_dec ** 2 / 2],
                         [0, 0, 0, T_dec ** 3 / 2, T_dec ** 2, T_dec],
                         [0, 0, 0, T_dec ** 2 / 2, T_dec, 1]]).float()


"""
def F(T):

    F = torch.tensor([[1, T, 0.5*T**2, 0, 0, 0],
                          [0, 1, T, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 1, T, 0.5*T**2],
                          [0, 0, 0, 0, 1, T],
                          [0, 0, 0, 0, 0, 1]]).float()
    return F

def Q(T):
    T_dec = T
    Q = torch.tensor([[T_dec ** 4 / 4, T_dec ** 3 / 2, T_dec ** 2 / 2, 0, 0, 0],
                         [T_dec ** 3 / 2, T_dec ** 2, T_dec, 0, 0, 0],
                         [T_dec ** 2 / 2, T_dec, 1, 0, 0, 0],
                         [0, 0, 0, T_dec ** 4 / 4, T_dec ** 3 / 2, T_dec ** 2 / 2],
                         [0, 0, 0, T_dec ** 3 / 2, T_dec ** 2, T_dec],
                         [0, 0, 0, T_dec ** 2 / 2, T_dec, 1]]).float()
    return Q
"""

H_PAO = torch.tensor([[1, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 1]]).float()

R_PAO = torch.tensor([[3*10**(0), 0, 0, 0],
                      [0, 1*10**(-1), 0, 0],
                      [0, 0, 3*10**(0), 0],
                      [0, 0, 0, 1*10**(-1)]]).float()

H_PO = torch.tensor([[1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0]]).float()

R_PO = torch.tensor([[3*10**(0), 0],
                     [0, 3*10**(0)]]).float()

H_AO = torch.tensor([[0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1]]).float()

R_AO = torch.tensor([[1*10**(-1), 0],
                     [0, 1*10**(-1)]]).float()



# 5.9196 GPS noise
