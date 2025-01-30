import mysql.connector
import os
from dotenv import load_dotenv, dotenv_values
import yfinance as yf
import datetime as dt

load_dotenv()

# print(os.getenv("USER"))

mydb = mysql.connector.connect(
    host="localhost",
    user=os.getenv("USER"),
    password=os.getenv("PASSWORD"),
    database="stocks"
)

cursor = mydb.cursor()

def get_data(stock, start, end):
    stockData = yf.download(stock, start=start, end=end)
    stockData = stockData['Close']

    # Insert data into the database
    for date, close_price in stockData.items():
        cursor.execute("""
            INSERT INTO stock_data (symbol, date, close_price)
            VALUES (%s, %s, %s)
        """, (stock, date, close_price))

    mydb.commit()
    # returns = stockData.pct_change()
    # meanReturns = returns.mean()
    # covMatrix = returns.cov()
    
    # return meanReturns, covMatrix

endDate = dt.datetime.now()
endDateString = endDate.strftime('%Y-$m-$d')
startDate = endDate - dt.timedelta(days=300)
startDateString = startDate.strftime('%Y-$m-$d')

get_data('AAPL', startDateString, endDateString)