import math
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from sympy import *
import datetime as dt
import pandas as pd
import pandas_datareader as pdr
from pandas_datareader import data as pdr
import yfinance as yf


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


# K = 220
# r = 0.05
# sigma = 0.32

# S_range = np.linspace(100, 300, 100)
# T_range = np.linspace(0.01, 1, 100)
# S, T = np.meshgrid(S_range, T_range)
# t = 0

# theta_values = np.vectorize(theta)(S, K, T, t, r, sigma)

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# # z = np.linspace(0, 1, 100)
# # x = z * np.sin(25 * z)
# # y = z * np.cos(25 * z)
# ax.plot_surface(S, T, theta_values)

# ax.set_xlabel('S')
# ax.set_ylabel('T-t')
# ax.set_zlabel('theta')

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


def get_data(stock, start, end):
    stockData = yf.download(stock, start=start, end=end)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

def print_summary(portfolio_sims, label):
    final_portfolio_values = portfolio_sims[-1, :]
    mean_final_value = np.mean(final_portfolio_values)
    var_final_value = np.var(final_portfolio_values)
    quart_25 = np.percentile(final_portfolio_values, 25)
    quart_75 = np.percentile(final_portfolio_values, 75)

    # 95% confidence that the portfolio will not get below VaR_95 in the given time period
    VaR_95 = np.percentile(final_portfolio_values, 5)
    VaR_99 = np.percentile(final_portfolio_values, 1)
    
    print(f"Summary for {label}:")
    print(f"1st Quartile: ${quart_25:,.2f}")
    print(f"Mean Final Value: ${mean_final_value:,.2f}")
    print(f"3rd Quartile: ${quart_75:,.2f}")
    print(f"Variance of Final Value: {var_final_value:,.2f}")    
    print(f"Value at Risk (95% confidence): ${VaR_95:,.2f}")
    print(f"Value at Risk (99% confidence): ${VaR_99:,.2f}")

def monte_carlo(stocks, start, end, weights, mc_sims, T, initialPortfolio):
    """
    Construct a predictive model of a porfolio

    Args:
        stocks (array): names of each stock
        start, end: start and end dates
        weights (array): proportion of each stock
        mc_sims (int): number of simulations to run
        T (int): time frame in days
        initialPortfolio (float): initial portfolio value

    Returns:
        Stats: quartiles, std dev, value at risk
        Graph

    """

    meanReturns, covMatrix = get_data(stocks, start, end)

    weights /= np.sum(weights)
    # print("weights:" + weights)

    meanMat = np.full(shape=(T, len(weights)), fill_value=meanReturns)
    meanMat = meanMat.T

    portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)

    for m in range(0, mc_sims):
        Z = np.random.normal(size=(T, len(weights)))
        L = np.linalg.cholesky(covMatrix)
        dailyReturns = meanMat + np.inner(L, Z)
        portfolio_sims[:,m] = np.cumprod(np.inner(weights, dailyReturns.T) + 1) * initialPortfolio
    
    percentiles = np.percentile(portfolio_sims, [25, 50, 75], axis=1)

    plt.plot(portfolio_sims, color='grey')

    plt.plot(percentiles[0, :], color='blue', label='25th Percentile')
    plt.plot(percentiles[1, :], color='green', label='50th Percentile')
    plt.plot(percentiles[2, :], color='red', label='75th Percentile')

    plt.ylabel('Portfolio Value ($)')
    plt.xlabel('Days')
    plt.title('MC simulation of a stock portfolio')

    # let = plt.legend()

    label = []

    for i in range(0, len(weights)):
        label.append([stocks[i], weights[i]])

    print_summary(portfolio_sims, label)
    plt.show()

endDate = dt.datetime.now() - dt.timedelta(days=365)
startDate = endDate - dt.timedelta(days=300)
# monte_carlo(['NVDA', 'INTC', 'SNAP', 'GOOGL'], startDate, endDate, [200, 200, 200, 1], 1000, 365, 10000)
monte_carlo(['AAPL', 'COST', 'TSM', 'TSLA', 'BAC', 'MSFT', 'AMZN', 'HD', 'KO', 'CMG'], 
            startDate, endDate, [3, 1, 1, 2, 1, 4, 2, 1, 1, 1], 1000, 365, 453153)

# data = yf.download(["AAPL","COST", 'TSM', 'TSLA', 'BAC', 'MSFT', 'AMZN', 'HD', 'KO', 'CMG'], start='2019-09-10', end='2020-10-09')
# data = data['Close']
# print(data.head())
# monte_carlo(['AAPL', 'COST', 'TSM'], 
#             startDate, endDate, [300, 100, 100], 1000, 365, 453153)