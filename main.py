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

# print(black_scholes(31.55, 22.75, 3.5, 0, 0.05, 0.5, 'call'))

# greeks for call options

def gamma(S, K, T, t, r, sigma):
    d_plus = d_p(S, K, T, t, r, sigma)
    return math.exp(-1 * d_plus ** 2 / 2) * 1 / (S * sigma * math.sqrt(2 * math.pi * (T - t)))

def theta(S, K, T, t, r, sigma):
    d_plus = d_p(S, K, T, t, r, sigma)
    d_minus = d_m(S, K, T, t, r, sigma)
    term1 = math.exp(-1 * d_plus ** 2 / 2) * -1 * S * sigma / (2 * math.sqrt(2 * math.pi * (T - t)))
    # term1 = - (S * st.norm.pdf(d_plus) * sigma) / (2 * math.sqrt(T - t))
    term2 = r * K * math.exp(-r * (T - t)) * st.norm.cdf(d_minus)
    return term1 - term2

def vega(S, K, T, t, r, sigma):
    d_plus = d_p(S, K, T, t, r, sigma)
    return S * math.sqrt(T - t) / math.sqrt(2 * math.pi) * math.exp(-1 * d_plus ** 2 / 2)

def rho(S, K, T, t, r, sigma):
    d_minus = d_m(S, K, T, t, r, sigma)
    return (T - t) * K * math.exp(-r * (T - t)) * st.norm.cdf(d_minus)


K = 220
r = 0.05
sigma = 0.32

S_range = np.linspace(100, 300, 100)
T_range = np.linspace(0.01, 1, 100)
S, T = np.meshgrid(S_range, T_range)
t = 0

theta_values = np.vectorize(theta)(S, K, T, t, r, sigma)

fig = plt.figure()
ax = plt.axes(projection='3d')
# z = np.linspace(0, 1, 100)
# x = z * np.sin(25 * z)
# y = z * np.cos(25 * z)
ax.plot_surface(S, T, theta_values)

ax.set_xlabel('S')
ax.set_ylabel('T-t')
ax.set_zlabel('theta')

plt.show()

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
