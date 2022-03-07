import torch
import src.parameters as params

T = params.T_cv
T_dec = T*params.ratio_cv
q = params.q_cv


#######################
### DATA GENERATION ###
#######################
F_gen = torch.tensor([[1, T, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, T],
                      [0, 0, 0, 1]]).float()

Q_gen = torch.tensor([[0, 0, 0, 0],
                      [0, 0, 0 ,0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]]).float()

H_gen = torch.tensor([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]]).float()

R_gen = torch.tensor([[0, 0, 0, 0],
                      [0, 0, 0 ,0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]]).float()

########################
### FULL OBSERVATION ###
########################
F_FO = torch.tensor([[1, T_dec, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, T_dec],
                     [0, 0, 0, 1]]).float()

Q_FO = torch.tensor([[T_dec**3/3, T_dec**2/2, 0, 0],
                     [T_dec**2/2, T_dec, 0, 0],
                     [0, 0, T_dec**3/3, T_dec**2/2],
                     [0, 0, T_dec**2/2, T_dec]]).float()

H_FO = torch.tensor([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]).float()

R_FO = torch.tensor([[1, 0, 0 ,0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]).float()

#######################
### POS OBSERVATION ###
#######################
F_PO = torch.tensor([[1, T_dec, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, T_dec],
                     [0, 0, 0, 1]]).float()

Q_PO = torch.tensor([[T_dec**3/3, T_dec**2/2, 0, 0],
                     [T_dec**2/2, T_dec, 0, 0],
                     [0, 0, T_dec**3/3, T_dec**2/2],
                     [0, 0, T_dec**2/2, T_dec]]).float()


H_PO = torch.tensor([[1, 0, 0, 0],
                     [0, 0, 1, 0]]).float()

R_PO = torch.tensor([[1, 0],
                      [0, 1]]).float()

#######################
### VEL OBSERVATION ###
#######################

F_VO = torch.tensor([[1, T_dec, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, T_dec],
                     [0, 0, 0, 1]]).float()

Q_VO = torch.tensor([[T_dec**3/3, T_dec**2/2, 0, 0],
                     [T_dec**2/2, T_dec, 0, 0],
                     [0, 0, T_dec**3/3, T_dec**2/2],
                     [0, 0, T_dec**2/2, T_dec]]).float()

H_VO = torch.tensor([[0, 1, 0, 0],
                     [0, 0, 0, 1]]).float()

R_VO = torch.tensor([[0.001, 0],
                      [0, 0.001]]).float()






