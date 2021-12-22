import torch
#######################
### Size of DataSet ###
#######################

# Number of Training Examples
#N_E = 20000
# Number of Cross Validation Examples
#N_CV = 100

# Number of Test Examples
#N_T = 10

###############################
### Network Hyperparameters ###
###############################
# Number of Training Epochs
N_Epochs = 10

# Number of Samples in Batch
N_B = 30
# Learning Rate
learning_rate = 1e-4

# L2 Weight Regularization - Weight Decay
wd = 1e-4

# Number of GRU layers
nGRU = 1

weights = torch.tensor([[1],[1],[1],[1],[1]])