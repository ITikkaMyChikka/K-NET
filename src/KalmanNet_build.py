from KalmanNet_nn import KalmanNetNN

def NNBuild(SysModel, preprocess_function, nGRU):

    Model = KalmanNetNN()

    Model.InitSystemDynamics(SysModel.f, SysModel.h, 2, 2)
    Model.InitSequence(SysModel.m1x_0, SysModel.m2x_0, SysModel.T)

    # Preprocessing function
    Model.preprocess = preprocess_function
    # Number of neurons in the 1st hidden layer
    H1_KNet = (2 + 2) * (10) * 8

    # Number of neurons in the 2nd hidden layer
    H2_KNet = (2 * 2) * 1 * (10)

    Model.InitKGainNet(H1_KNet, H2_KNet, nGRU)

    return Model