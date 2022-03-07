Ts = 0.00005  # 0.5ms Sampling time in sec
T = 0.0005  # 5ms sampling
perturbation = Ts
observation_noise = T
ratio = int(T/Ts)


q_cv = 1*10**(7)  #1*10**(-2) # variance sigma^2
T_cv = 5*10**(-5)  # 50 microsecond
ratio_cv = 1000
perturbation_cv = 5*10**(-6)


# Average acceleration is between 3-4m/s^2
q_ca = 7*10**(-5)  # this is a squared value, i.e the variance
T_ca = 5*10**(-5)  # 50 microsecond
ratio_ca = 1000
perturbation_ca = 5*10**(-6)


# Average acceleration is between 3-4m/s^2
q_ctra = 7*10**(-5)  # this is a squared value, i.e the variance
T_ctra = 5*10**(-5)  # 50 microsecond
ratio_ctra = 10
perturbation_ctra = 5*10**(-6)
sa = 0.0008  # max acceleration longitudinal
sw = 0.0002  # max steering angle in Hz
sa_gen = sa/ratio_ctra  # max acceleration longitudinal
sw_gen = sw/ratio_ctra  # max steering angle in Hz
