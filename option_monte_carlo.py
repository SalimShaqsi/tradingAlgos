import numpy as np
import math
import matplotlib.pyplot as plt


def mc_price(S0, K, T, r, sigma, Nt, Np):
    dt = T/Nt
    logS0 = math.log(S0, math.exp(1))
    Z = np.random.standard_normal((Nt-1, Np))
    dlogS = (r-sigma**2/2) * dt + sigma * math.sqrt(dt) * Z
    logS = np.vstack((np.full((1, Np), logS0), dlogS)).cumsum(axis=0)
    S = np.exp(logS)
    S_T = S[-1, :]
    V_T = math.exp(-r*T) * (S_T - K) * (S_T > K)

    plt.plot(S[:, :100])
    plt.show()

    return V_T.mean()


S0 = 100
K = 100
T = 1
r = 0.06
sigma = 0.15
Nt = 252
Np = 100_000

print(mc_price(S0, K, T, r, sigma, Nt, Np))

# scales = [1,2,4,8,16,32,64,128,256,512,1028]
#
# Nt_sens = [mc_price(S0, K, T, r, sigma, Nt*i, 1000, plot=False) for i in scales]
# #Np_sens = [mc_price(S0, K, T, r, sigma, 252, Np*i, plot=False) for i in scales]
#
#
# plt.plot(Nt_sens)
# plt.show()
