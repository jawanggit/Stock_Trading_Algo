from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import csv
import talib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import sklearn.model_selection as cv
import sklearn.datasets as datasets
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


def PullData(stock, enddate):

    API_key = '6JLJYIR4IV3WWRWS'
    ts = TimeSeries(key = API_key,output_format='pandas')
    data = ts.get_daily_adjusted(stock,outputsize = 'full') #Stock
    df = data[0].sort_index()
    df = df.append(pd.Series(name=pd.to_datetime(enddate))) #'2021-03-29'

    return df

def AddIndicators(df_final, open_price):

    df_final['yest_volume'] = df_final['volume'].shift(periods = 1)
    df_final['yest_low'] = df_final['low'].shift(periods = 1)
    df_final['yest_high'] = df_final['high'].shift(periods = 1)
    df_final['yest_close'] = df_final['close'].shift(periods = 1)
    df_final['yest_open'] = df_final['open'].shift(periods = 1)
    df_final['yest_RSI'] = df_final['RSI'].shift(periods = 1)
    df_final['yest_CMO'] = df_final['CMO'].shift(periods = 1)
    df_final['yest_MA'] = df_final['Moving Average'].shift(periods = 1)
    df_final['yest_MACD'] = df_final['MACD'].shift(periods = 1)
    df_final['yest_MACD_signal'] = df_final['MACD_signal'].shift(periods = 1)
    df_final['yest_ROC'] = df_final['ROC'].shift(periods = 1)
    df_final['yest_PPO'] = df_final['PPO'].shift(periods = 1);

    df_final.iloc[-1,0] = open_price
    df_final.iloc[-1] = df_final.iloc[-1].fillna(0)
    
    df_final.dropna(axis = 0, inplace = True)
    df_model = df_final.drop(columns=['volume','low','high','close','RSI','CMO','Moving Average','MACD','MACD_signal','ROC','PPO']);

    return df_model




