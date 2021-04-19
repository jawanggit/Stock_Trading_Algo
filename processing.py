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
import math

def PullHourlyData(stock):

    API_key = 'CSMN0LYTQ5UYMVUT'#'6JLJYIR4IV3WWRWS'
    ts = TimeSeries(key = API_key,output_format='csv')
    final_df = pd.DataFrame


    dct = {}
    for i,slices in enumerate(['year1month9','year1month8','year1month7','year1month6','year1month5','year1month4','year1month3','year1month2','year1month1']):
     
        data = ts.get_intraday_extended('DAL',interval='60min',slice  = slices)
        dct[i] = pd.DataFrame(list(data[0]))
       
    final_df = pd.concat([dct[0],dct[1],dct[2],dct[3],dct[4],dct[5],dct[6],dct[7],dct[8]], ignore_index=True)
    columns = final_df.iloc[0]; columns
    
    final_df = final_df[final_df.iloc[:,0] != 'time']
    final_df = final_df.rename(columns = {0:'time',1:'open',2:'high',3:'low',4:'close',5:'volume'})
    
    final_df['time'] = pd.to_datetime(final_df['time'])
    final_df = final_df.set_index('time')
    final_df = final_df.sort_index()
    
    for col in final_df.columns:
        final_df[col]=pd.to_numeric(final_df[col])
    
    ts_low = final_df['low']
    ts_high = final_df['high']
    ts_low = ts_low.asfreq('H', method = 'ffill')
    ts_high = ts_high.asfreq('H', method = 'ffill')
    ts_low = ts_low.apply(lambda x:math.log(x))
    ts_high = ts_high.apply(lambda x:math.log(x))
    ts_low.to_csv('hourly_stock_data.csv',header = True)
    ts_high.to_csv('hourly_stock_data_high.csv',header = True)
    

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

    df_final.to_csv('stock_data.csv')
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
    df_model = df_final.drop(columns=['volume','low','high','close',
    'RSI','CMO','Moving Average','MACD','MACD_signal','ROC','PPO'])

    df_final = df_final.iloc[-3000:,:]
    #df_model has all features shifted by 1 day and the actual day's open price"
    df_model = df_model.iloc[-3000:,:]
    return df_model , df_final

def HighLowTimestamp(df_final):

    API_key = '6JLJYIR4IV3WWRWS'
    ts = TimeSeries(key = API_key,output_format='csv')
    ### daily chart, include percent change ###

    dct = {}
    for i,slices in enumerate(['year1month3','year1month2','year1month1']):
    
    
        data = ts.get_intraday_extended('DAL',interval='60min',slice  = slices)
        dct[i] = pd.DataFrame(list(data[0]))
            #     data = ts.get_intraday_extended('DAL',interval='60min',slice  = slices)
            # df.append(pd.DataFrame(list(data[0])),ignore_index = True)
            # print(slices)
    
    final_df = pd.concat([dct[0],dct[1],dct[2]], ignore_index=True)
    columns = final_df.iloc[0]; columns
    final_df = final_df[final_df.iloc[:,0] != 'time']
    final_df = final_df.rename(columns = {0:'time',1:'open',2:'high',3:'low',4:'close',5:'volume'})
    final_df['time'] = pd.to_datetime(final_df['time'])
    df = (final_df.set_index('time')
            .between_time('10:00:00', '16:00:00')
            .reset_index()
            .reindex(columns=final_df.columns))

    df['Date'] = pd.to_datetime(df['time']).dt.date
    df['Timestamp'] = pd.to_datetime(df['time']).dt.time
    f_df = df

    date_lst = set(f_df['Date'])
    time_dict = {}
    for date in date_lst:
        day = f_df[f_df['Date']==date]
        high_value = day.sort_values('high', ascending=False)['Timestamp'].iloc[0]
        time_dict[str(date)] = {'high':high_value}
        low_value = day.sort_values('low')['Timestamp'].iloc[0]
        time_dict[str(date)]['low'] = low_value

    return time_dict


#LSTM transformation

def lstm_data_transform(x_data, low_actual_data, high_actual_data,  num_steps=5):
    """ Changes data to the format for LSTM training for sliding
     window approach (ref: https://towardsdatascience.com/
     how-to-reshape-data-and-do-regression-for-time-series-
     using-lstm-133dad96cd00) """
    # Prepare the list for the transformed data
    X, low_actual, high_actual = list(), list(), list()
    # Loop of the entire data set
    for i in range(x_data.shape[0]):
        # compute a new (sliding window) index
        end_ix = i + num_steps
        # if index is larger than the size of the dataset, we stop
        if end_ix >= x_data.shape[0]:
            break
        # Get a sequence of data for x
        seq_X = x_data[i:end_ix]
        # Get only the last element of the sequency for y
        seq_low_actual = low_actual_data[end_ix]
        seq_high_actual = high_actual_data[end_ix]
        
        # Append the list with sequencies
        X.append(seq_X)
        low_actual.append(seq_low_actual)
        high_actual.append(seq_high_actual)
    # Make final arrays
    x_array = np.array(X)
    low_array = np.array(low_actual)
    high_array = np.array(high_actual)
    
    return x_array, low_array, high_array

