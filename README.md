# Stock_Trading_Algo
Stock Trading Algorithm that uses Machine Learning to calculate entry and exist points for daily trades


# Outline:
1. Business Problem:

Predicting the price of stocks accurately is a critical business problem for investment banks and hedge fund managers. This allows them to generate reutrns for their customers which in turn results in greater and greater investments and commissions for these companies.

2. Goal of Project:
Using various supervised machine learning models, I want to create my own trading algorithm and measure its performance/returns on various stocks over a 3 month period. I will compare these returns to a simple "Buy and Hold" strategy for a 3 month period.

3. Dataset

The stock data for my project was obtained using the Alpha Vantage API. I am pulling the following information:

Daily Open
Daily High
Daily Low
Daily Close
Daily Volume

I am generating the following technical indicators which are additional features in the data for the model to use:

MACD
ROC
PPO
RSI

4. Data Pre-Processing


5. Machine Learning Algorithms that were Applied

    - ARIMA
    - Gradient Boosting
    - RNN


6. Testing 

    - Testing window was 3 months for the following stocks.
    - Baseline Strategy of Buy and Hold
    - ML-Based Trading Strategy

    
7. Results
    - Stocks returns 

