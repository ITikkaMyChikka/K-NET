import torch


# Number of Training Examples
N_E = 1

# Number of Cross Validation Examples
N_CV = 1

# Number of Testing Examples
N_T = 1


def DataGen(SysModel_data, fileName, T_test,randomInit=False):

    """
    ##################################
    ### Generate Training Sequence ###
    ##################################
    SysModel_data.GenerateBatch(N_E, T, randomInit=randomInit)
    training_input = SysModel_data.Input  # self.y observations
    training_target = SysModel_data.Target  # self.x states
    print("Training data generated")
    ####################################
    ### Generate Validation Sequence ###
    ####################################
    SysModel_data.GenerateBatch(N_CV, T, randomInit=randomInit)
    cv_input = SysModel_data.Input
    cv_target = SysModel_data.Target
    print("Validation data generated")
    """
    ##############################
    ### Generate Test Sequence ###
    ##############################
    SysModel_data.GenerateBatch(N_T, T_test, randomInit=randomInit)
    test_input = SysModel_data.Input
    test_target = SysModel_data.Target
    print("Test data generated")

    #################
    ### Save Data ###
    #################
    torch.save([test_input, test_target], fileName)

