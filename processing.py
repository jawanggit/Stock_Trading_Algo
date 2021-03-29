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


    close = df['4. close'] # switched to using 'close' price rather than 'adjusted close' price for TAs
    close = np.array(close)
    df['RSI'] = talib.RSI(close);
    df['CMO'] = talib.CMO(close, timeperiod=14)
    df['Moving Average'] = talib.MA(close,timeperiod = 20)
    df['MACD'],df['MACD_signal'] = talib.MACD(close,fastperiod =12, slowperiod =26, signalperiod = 9)[0:2]
    df['ROC']= talib.ROC(close, timeperiod=10)
    df['PPO'] = talib.PPO(close, fastperiod=12, slowperiod=26, matype=0)
    df_final = df.drop(columns=['5. adjusted close','7. dividend amount', '8. split coefficient'])
    df_final = df_final.rename(columns = {'1. open': 'open','2. high': 'high','3. low':'low', '4. close':'close','6. volume':'volume'})

    return df_final

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

    return df_model , df_final




