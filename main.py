import math
import numpy as np
import scipy.stats as st

def black_scholes(S, K, T, t, r, sigma, option_type = 'call'):
    
    d_plus = ( np.log( S / K ) + ( T - t ) * ( r + sigma ** 2 / 2 ) ) / ( sigma * math.sqrt( T - t ) )
    d_minus = d_plus - sigma * math.sqrt( T - t )
    
    if option_type == 'call':
        return S * st.norm.cdf(d_plus) - K * math.exp( -r * ( T - t ) ) * st.norm.cdf(d_minus)
    elif option_type == 'put':
        return -S * st.norm.cdf(-d_plus) + K * math.exp( -r * ( T - t ) ) * st.norm.cdf(-d_minus)


calc_type = input("Calculator type: \n Press 1 for Black-Scholes Calculation \n Press 2 for\n")

if calc_type == "1":
    S = float(input("Input the prince of the underlying stock (S):\n"))
    K = float(input("Input the strike price (K):\n"))
    T = float(input("Input the exercise date in years (T):\n"))
    t = float(input("Input the current time (t):\n"))
    r = float(input("Input the risk-free rate (r):\n"))
    sigma = float(input("Input the volatility of the stock (sigma):\n"))
    option_type = input("Input the option type (call/put):\n")
    print("Calculating...")
    print("The price of the European", option_type, "option at time", t, "is", black_scholes(S, K, T, t, r, sigma, option_type))

else:
    print("not done yet")