# Stock_Trading_Algo
Stock Trading Algorithm that uses Machine Learning to calculate entry and exist points for daily trades


# Outline
## 1. Business Problem:

Algorithmic trading applies machine learning models to predict when to sell and buy stocks in the stock market. Generating accurate predictions or forecasts for stock prices and other financial instruments is the central business problem for all investment banks and hedge fund managers. This project seeks to apply many of the Machine Learning algorithms that I have learned in order to generate a trading strategy that can yield returns over a 3 month period greater than simply investing in the SPY500 during the same period of time. 

Disclaimer - this project is entirely intended for research purposes and applying the algorithms and/or trading strategies outlined below is no guarantee of positive returns

## 2. Business and Data Understanding:

There are many metrics/technical indicators that can be applied to a machine learning model when it comes to stocks. Based on research I did on stock prediction algorithms (ref 1), I decided to use the following lagging technical indicators: Moving Average Convergence Divergence (MACD), Price Rate of Change (ROC), Percentage Price Oscillator (PPO), and Relative Strength Index (RSI). For my initial round of models, I used the standard number of periods that are used to calculate these indicators.

Additionally, each row of data included the prior trading day's OPENING, VOLUME, LOW, HIGH, and CLOSING price along with the current trading day's OPENING price. In this way, there would be no data leakage when predicting the current day's LOW and HIGH price since the algorithm would ideally run once the stock market opened each day. The exact details of the trading strategy will be outlined in greater detail in the Evaluation section.

## 3. Data Preparation

There are numerous options for pulling stock data. I chose to use the AlphaVantage API to pull the opening, closing, daily high, and daily low price. To calculate the technical indicators, I used the open-source TA-Lib package which has built-in methods for calculating the MACD, ROC, PPO, and RSI.

The processing.py file contains the two main functions for the data preparation. The "PullHourlyData", "PullData" and "AddIndicators()" functions pull over 10 years of historical daily stock data and build the technical indicators using the TA-Lib package, respectively.

## 4. Creating the Machine Learning (ML) Models

### AutoRegressive Integrated Moving Average (ARIMA) Model:

The first ML model I built was an ARIMA model using the statsmodels package. Since stock prices are type of timeseries data, using an ARIMA model seemed like an ideal first step in predicting/forecasting future prices.

The ARIMA model did not utilize the technical indicators/additional features that I generated and relies strictly on price and time to generate its predictions. The stock I choose to forecast on was Delta Airlines (stock ticker: DAL) and in order to apply an ARIMA model to it, I needed to remove the upward trend and slight seasonality (Fig A.) to make the timeseries stationary. By applying a log transformation to the prices and taking the difference between values, I was able to create a stationary timeseries that passed the Augumented Dickey-Fuller (ADF) test for stationarity.

To determine the autocorrelation parameter for the model, I examined the autocorrelation and partial autocorrelation plots. Based on these plots (Fig B.) and research done by Andrew Nau on choosing p and q values, I choose a p value of 2 and q value of 0. 

The predicted values for the low price were extremely high, resulting in a high RMSE. Due to fact that I am predicting an hourly low for 60 trading days, the time required for forecasting additional ARIMA models did not make it feasible to explore ARIMA models further.


### Gradient Boosting Regression Model (GBR):

The second model I built was a GBR model. I choose to use a GBR model because research I had done into machine learning-based stock trading often mentioned and utilized GBR to due similar prediction on stocks. Because GBR is a tree-based model, I did not need to standardize the dataset or apply any transformations to the dataset. I was also able to leverage all the additional features that I created using technical indicators which offered a much richer dataset to apply to a GBR model.

To tune this model, I decided to focus on the learning rate, n_estimators, and the regulurization value (alpha) using a quantile loss function. I choose to use a quantile loss function for the model because I was predicting high and low daily prices and I wanted to penalize the predictions differently based on whether or not the model was predicting a daily high or low price. 

Based off a custom GridSearch that minimized RMSE values for both the alpha_low and alpha_high values, I determined an inital value of .3 for alpha_low and .6 for alph_high to test in the trading strategy. The predicted values for the high and low each day are plotted over the actual high and low (Fig C). One can see that these predicted values do a relatively good job matching the actual values.


### Evaluation:

The ultimate test for these models is to calculate the returns from these predictions. With an ARIMA model, the model  



