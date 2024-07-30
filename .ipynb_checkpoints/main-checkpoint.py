import math
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from sympy import *


def d_p(S, K, T, t, r, sigma):
    return ( np.log( S / K ) + ( T - t ) * ( r + sigma ** 2 / 2 ) ) / ( sigma * math.sqrt( T - t ) )

def d_m(S, K, T, t, r, sigma):
    d_plus = d_p(S, K, T, t, r, sigma)
    return d_plus - sigma * math.sqrt( T - t )

# print(d_p(50, 50, 90/365, 0, 0.05, 0.4))

def black_scholes(S, K, T, t, r, sigma, option_type = 'call'):

    d_plus = d_p(S, K, T, t, r, sigma)
    d_minus = d_m(S, K, T, t, r, sigma)
    
    if option_type == 'call':
        return S * st.norm.cdf(d_plus) - K * math.exp( -r * ( T - t ) ) * st.norm.cdf(d_minus)
    elif option_type == 'put':
        return -S * st.norm.cdf(-d_plus) + K * math.exp( -r * ( T - t ) ) * st.norm.cdf(-d_minus)

print(black_scholes(31.55, 22.75, 3.5, 0, 0.05, 0.5, 'call'))


# fig = plt.figure()
# ax = plt.axes(projection='3d')
# z = np.linspace(0, 1, 100)
# x = z * np.sin(25 * z)
# y = z * np.cos(25 * z)
# ax.plot3D(x, y, z, 'green')
# plt.show()





# calc_type = input("Calculator type: \n Press 1 for Black-Scholes Calculation \n Press 2 for\n")

# if calc_type == "1":
#     S = float(input("Input the prince of the underlying stock (S):\n"))
#     K = float(input("Input the strike price (K):\n"))
#     T = float(input("Input the exercise date in years (T):\n"))
#     t = float(input("Input the current time (t):\n"))
#     r = float(input("Input the risk-free rate (r):\n"))
#     sigma = float(input("Input the volatility of the stock (sigma):\n"))
#     option_type = input("Input the option type (call/put):\n")
#     print("Calculating...")
#     print("The price of the European", option_type, "option at time", t, "is", black_scholes(S, K, T, t, r, sigma, option_type))

# else:
#     print("not done yet")
