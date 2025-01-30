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
import mysql.connector


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

    VaR_95 = np.percentile(final_portfolio_values, 5)
    VaR_99 = np.percentile(final_portfolio_values, 1)
    
    print(f"Summary for:")
    for l in label:
        print(f"  {l}")
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

    label = []

    for i in range(0, len(weights)):
        label.append([stocks[i], weights[i]])

    print(start)
    print_summary(portfolio_sims, label)
    plt.show()

def main():
    # monte_carlo(['NVDA', 'INTC', 'SNAP', 'GOOGL'], startDate, endDate, [200, 200, 200, 1], 1000, 365, 10000)

    # 2024-25; ended up ?
    endDate = dt.datetime.now()
    startDate = endDate - dt.timedelta(days=300)
    monte_carlo(['AAPL', 'COST', 'TSM', 'TSLA', 'BAC', 'MSFT', 'AMZN', 'HD', 'KO', 'CMG'], 
                startDate, endDate, [3, 1, 1, 2, 1, 4, 2, 1, 1, 1], 1000, 365, 453153)

    # 2023-24; ended up 453153
    endDate = dt.datetime.now() - dt.timedelta(days=365)
    startDate = endDate - dt.timedelta(days=300)
    monte_carlo(['AAPL', 'COST', 'TSM', 'TSLA', 'BAC', 'MSFT', 'AMZN', 'HD', 'KO', 'CMG'], 
                startDate, endDate, [3, 1, 1, 2, 1, 4, 2, 1, 1, 1], 1000, 365, 377378)

    # 2017-18; ended up 626318
    endDate = dt.datetime.now()  - dt.timedelta(days=365*7)
    startDate = endDate - dt.timedelta(days=300)
    monte_carlo(['AAPL', 'COST', 'TSM', 'TSLA', 'BAC', 'MSFT', 'AMZN', 'HD', 'KO', 'CMG'], 
                startDate, endDate, [3, 1, 1, 2, 1, 4, 2, 1, 1, 1], 1000, 365, 414988)

    # 2013-14; ended up 247644
    endDate = dt.datetime.now()  - dt.timedelta(days=365*11)
    startDate = endDate - dt.timedelta(days=300)

    monte_carlo(['AAPL', 'COST', 'TSM', 'TSLA', 'BAC', 'MSFT', 'AMZN', 'HD', 'KO', 'CMG'], 
                startDate, endDate, [3, 1, 1, 2, 1, 4, 2, 1, 1, 1], 1000, 365, 185892)



if __name__ == "__main__":
    main()