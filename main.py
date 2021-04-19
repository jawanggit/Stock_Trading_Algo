import pandas as pd
import math
import csv
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
import sklearn.model_selection as cv
import sklearn.datasets as datasets
import collections
from statsmodels.tsa.arima.model import ARIMA 
# import stats
from tensorflow import keras
from tensorflow.keras import layers
import processing
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def Overall_Return(self,results, return_type = 'optimal'):
    overall_return = 0
    trade_lst = []
    if return_type == 'optimal':
        for index, row in results.iterrows():
            if row['off_by'] >0:
                if row['actual_high'] >row['pred_high'] and row['high'] > row['low']:
                    trade_lst.append(row)
                    overall_return += row['return_plow_minus_phigh']
                    
                else:
                    trade_lst.append(row)
                    overall_return += row['return']
        
        trades  = pd.DataFrame(trade_lst,columns=results.columns)
        trades.to_excel('TRADES_GBR.xlsx')      
        return overall_return-results.iloc[-1,5] # removes most recent actual price which is neg val

if __name__ == '__main__':

    #Uncomment line below to get an up-to-date hourly pull of stock data
    #processing.PullHourlyData('DAL')
    df_hourly_low = pd.read_csv('hourly_stock_data.csv', index_col = 'time')
    df_hourly_high = pd.read_csv('hourly_stock_data_high.csv', index_col = 'time')
    
    #Uncomment line below to get an up-to-date daily pull of stock data
    #df = processing.PullData('DAL', '2021-04-6')
    df = pd.read_csv('stock_data.csv',index_col = 'date')
    df.index = pd.to_datetime(df.index)

    df_model , df_final = processing.AddIndicators(df, '50.30') #processing.py

    timestamp = pd.read_csv('timestamp_high_low.csv', index_col = 0)
    timestamp.index = pd.to_datetime(timestamp.index)
    
    #settings for generating results
    testing_mode = False
    RNN = True
    num_steps = 5
    GBR = False
    ARIMA = False
    test_window = 3
    return_type = 'optimal'
    
    #RNN
    if RNN:

        #split train/test
        train_ind = int(df_model.shape[0]*.8) #need to correct this for Jan 1,2021 start for test data
        x_train = df_model[:train_ind]
        x_test = df_model[train_ind:]
        low_y_train = df_final['low'].iloc[:train_ind]
        low_y_test = df_final['low'].iloc[train_ind:]
        high_y_train = df_final['high'].iloc[:train_ind]
        high_y_test = df_final['high'].iloc[train_ind:]
        
        #scaling data for input into neural network
        scaler_x = StandardScaler()
        scaler_low_y = StandardScaler()
        scaler_high_y = StandardScaler()
        
        x_train_sc = scaler_x.fit_transform(x_train)
        x_test_sc = scaler_x.transform(x_test)
        
        #reshape scaler_low_y since it is a row vector
        low_y_train_sc = scaler_low_y.fit_transform((low_y_train.values).reshape(-1,1))
        low_y_test_sc = scaler_low_y.transform((low_y_test.values).reshape(-1,1))
        
        high_y_train_sc = scaler_high_y.fit_transform((high_y_train.values).reshape(-1,1))
        high_y_test_sc = scaler_high_y.transform((high_y_test.values).reshape(-1,1))

        #transform 2D data to 3D array for LSTM nodes
        #train set
        (x_train_transformed,
         low_y_train_transformed, high_y_train_transformed) = processing.lstm_data_transform(x_train_sc,
                                                                  low_y_train_sc, high_y_train_sc)
        
        # test set
        (x_test_transformed,
        low_y_test_transformed, high_y_test_transformed) = processing.lstm_data_transform(x_test_sc,
                                                                 low_y_test_sc, high_y_test_sc)
             
        #compile model for low price
        model = keras.Sequential()
        model.add(layers.LSTM(100, activation='tanh', input_shape=(num_steps, 13), 
                    return_sequences=False))
        model.add(layers.Dense(units=50, activation='relu'))
        model.add(layers.Dense(units=1, activation='linear'))
        adam = keras.optimizers.Adam(lr=0.001)
        model.compile(optimizer=adam, loss='mse')
        
        #uncomment lines below to determine number of epochs:
        #history = model.fit(x_train_transformed, low_y_train_transformed, epochs=10, validation_data=(x_test_transformed,low_y_test_transformed)) 
        #generate train vs validation loss plot to determine epoch
        #processing.plot_train_vs_val_loss(history, 'Low_price_loss_graph.png')
        #predict validation data using optimal epoch from graph (3 epochs)

        model.fit(x_train_transformed, low_y_train_transformed, epochs=3)
        results = model.predict(x_test_transformed)
        final_results = scaler_low_y.inverse_transform(np.array(results))
        print(final_results)
        # evaluation = model.evaluate(x_test_transformed,low_y_test_transformed)
        # print(evaluation)

        processing.plot_val_vs_actual(final_results, low_y_test, 'Actual Low vs Predicted Plot.png')


        #compile mode for high price
        model = keras.Sequential()
        model.add(layers.LSTM(100, activation='tanh', input_shape=(num_steps, 13), 
                    return_sequences=False))
        model.add(layers.Dense(units=50, activation='relu'))
        model.add(layers.Dense(units=1, activation='linear'))
        adam = keras.optimizers.Adam(lr=0.001)
        model.compile(optimizer=adam, loss='mse')
        
        #uncomment lines below to the determine number of epochs:
        #history = model.fit(x_train_transformed, high_y_train_transformed, epochs=10, validation_data=(x_test_transformed,high_y_test_transformed))
        #generate train vs val loss plot
        #processing.plot_train_vs_val_loss(history, 'High_price_loss_graph.png')


    #Gradient Boosting
    if GBR:
        if testing_mode:          
            testing_GBR(alpha_l_list, alpha_h_list,test_window)
        else:            
            #test one scenario and generate trade history
            #GBR_test = GBR_()
            alpha_l = .3
            alpha_h = .7
            return_type = 'optimal'
        
            #generate daily low and high price prediction for each day in test window
            df_model_with_low,rmse_low = Daily_Low(df_model, alpha_l, test_window)
            results,rmse_high = Daily_High(df_model,df_model_with_low, alpha_h, test_window)
            
            #join results with timestamp for high and low daily price csv
            results['dates']=pd.to_datetime(results['dates'])
            results = results.set_index('dates')
            results = results.join(timestamp, how = 'left')
            results.to_excel('GBR_4_5.xlsx')

            #Overall return for test window assuming fixed number of shares purchased and sold during each buy and sell transaction
            value = Overall_Return(results,return_type = return_type)
            print(f'GBR return over {test_window}: {value}')
            print(f'GBR RMSE over {test_window}: {rmse_low}')
    

    #ARIMA
    if ARIMA:
        
        ARIMA_test = ARIMA_()
        df_low,rmse_low = ARIMA_test.Daily_Low(df_hourly_low,'1/1/2021', test_window)
        df_low.to_excel('ARIMA_low.xlsx')
        results = ARIMA_test.Daily_High(df_hourly_high,df_low,'1/1/2021',test_window)
        results.to_excel('ARIMA_full_results.xlsx')

        print(f'ARIMA return over 60 trading days: {ARIMA_test.Overall_Return(results)}')
        print(f'ARIMA RMSE: {rmse_low}')

