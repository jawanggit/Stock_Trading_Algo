import pandas as pd
import math
import csv
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
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

def Overall_Return(results, return_type = 'optimal'):
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
        trades.to_excel('TRADES_RNN.xlsx')      
        return overall_return# removes most recent actual price which is neg val

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
    num_steps = (20,20)
    epochs = (1,1)
    experiment = 1
    
    GBR = False
    ARIMA = False
    test_window = 3
    return_type = 'optimal'
    
    #RNN
    if RNN:
                   
        #split train/test
        timestart = pd.to_datetime('2021-01-04')
        print(timestart)
        low_index_start = df_model.index.get_loc(timestart)-num_steps[0]
        high_index_start = df_model.index.get_loc(timestart)-num_steps[1]     
    
        # masklow_train = df_model.iloc[0:low_index_start]
        # masklow_test = df_model.iloc[:low_index_start]
        # maskhigh_train = df_model.index < high_index_start
        # maskhigh_test = df_model.index >= high_index_start

        x_train_low = df_model.iloc[0:low_index_start]
        # x_train_low.iloc[-10:].to_excel("x_train.xlsx")
        x_test_low = df_model.iloc[low_index_start:]
        # x_test_low.iloc[-10:].to_excel("x_test.xlsx")
        x_train_high = df_model.iloc[0:high_index_start]
        x_test_high = df_model.iloc[high_index_start:]
        
        
        low_y_train = df_final['low'].iloc[0:low_index_start]
        # low_y_train.iloc[-10:].to_excel("low_y.xlsx")
        low_y_test = df_final['low'].iloc[low_index_start:]
        # low_y_test.iloc[-10:].to_excel("y_test.xlsx")
        high_y_train = df_final['high'].iloc[0:high_index_start]
        high_y_test = df_final['high'].iloc[high_index_start:]
        
        #scaling data for input into neural network
        scaler_low_x = StandardScaler()
        scaler_high_x = StandardScaler()
        scaler_low_y = StandardScaler()
        scaler_high_y = StandardScaler()
        
        low_x_train_sc = scaler_low_x.fit_transform(x_train_low)
        high_x_train_sc = scaler_high_x.fit_transform(x_train_high)
             
        low_x_test_sc = scaler_low_x.transform(x_test_low)
        high_x_test_sc = scaler_high_x.transform(x_test_high)
        
        #reshape scaler_low_y since it is a row vector
        low_y_train_sc = scaler_low_y.fit_transform((low_y_train.values).reshape(-1,1))
        low_y_test_sc = scaler_low_y.transform((low_y_test.values).reshape(-1,1))
        
        high_y_train_sc = scaler_high_y.fit_transform((high_y_train.values).reshape(-1,1))
        high_y_test_sc = scaler_high_y.transform((high_y_test.values).reshape(-1,1))

        #transform 2D data to 3D array for LSTM nodes
        #train set
        (low_x_train_transformed, low_y_train_transformed) = processing.lstm_data_transform(low_x_train_sc,
                                                    low_y_train_sc, num_steps[0])
        (high_x_train_transformed, high_y_train_transformed) = processing.lstm_data_transform(high_x_train_sc,
                                                    high_y_train_sc, num_steps[1])
        
        # test set
        (low_x_test_transformed, low_y_test_transformed) = processing.lstm_data_transform(low_x_test_sc,
                                                    low_y_test_sc, num_steps[0])
        (high_x_test_transformed, high_y_test_transformed) = processing.lstm_data_transform(high_x_test_sc,
                                                    high_y_test_sc, num_steps[1])
    
        error_scores = list()
        experiment_results = pd.DataFrame()

        for r in range(experiment):

            #compile model for low price
            model = keras.Sequential()
            model.add(layers.LSTM(150, activation='tanh', batch_input_shape=(1, num_steps[0], 13), 
                        return_sequences=False))
            model.add(layers.Dropout(.2))
            model.add(layers.Dense(units=128, activation='relu'))
            model.add(layers.Dense(units=64, activation='relu'))
            model.add(layers.Dense(units=32, activation='relu'))
            model.add(layers.Dense(units=8, activation='relu'))
            model.add(layers.Dense(units=1, activation='linear'))

            print(model.summary())
            
            adam = keras.optimizers.Adam(lr=0.0001)
            model.compile(optimizer=adam, loss='mse')
            
            # # uncomment lines below to determine number of epochs:
            # history = model.fit(low_x_train_transformed, low_y_train_transformed, epochs = epochs[0],
            #              validation_data=(low_x_test_transformed,low_y_test_transformed)) 
            # # generate train vs validation loss plot to determine epoch
            # processing.plot_train_vs_val_loss(history, 'Low_price_loss_graph.png')
            # # predict validation data using optimal epoch from graph (3 epochs)

            model.fit(low_x_train_transformed, low_y_train_transformed, batch_size=1, epochs = epochs[0], verbose = 0, shuffle = False)
            results = model.predict(low_x_test_transformed, batch_size=1)  
            final_results = scaler_low_y.inverse_transform(np.array(results))
            
            rmse = sqrt(mean_squared_error(low_y_test.iloc[num_steps[0]:],final_results))
            print(f'Test RMSE: {r+1} : {rmse}')
            experiment_results['predictions'] = final_results.flatten()
            error_scores.append(rmse)

        experiment_results['mean']= experiment_results.mean(axis=1)
        experiment_results.to_csv('RNN_low_predictions.csv', index = False)

        df_results = pd.DataFrame()
        df_results['results'] = error_scores
        print(df_results.describe())
        df_results.to_csv('RNN_low_RMSE_experiment_fixed.csv',index = False)
        
        # evaluation = model.evaluate(low_x_test_transformed,low_y_test_transformed)
        # print(evaluation)

        processing.plot_val_vs_actual(experiment_results['mean'], low_y_test,
                                     'Actual Low vs Predicted Plot.png', 'Low',num_steps[0])
        final_df_results = pd.DataFrame(experiment_results['mean'][:-1],index = low_y_test.iloc[num_steps[0]:-1].index,
                                     columns = ['pred_low'])
        for r in range(experiment):
            #compile mode for high price

            model = keras.Sequential()
            model.add(layers.LSTM(150, activation='tanh', batch_input_shape=(1, num_steps[1], 13), 
                    return_sequences=False))
            model.add(layers.Dropout(.2))
            model.add(layers.Dense(units=128, activation='relu'))
            model.add(layers.Dense(units=64, activation='relu'))
            model.add(layers.Dense(units=32, activation='relu'))
            model.add(layers.Dense(units=8, activation='relu'))
            model.add(layers.Dense(units=1, activation='linear'))

            print(model.summary())
            adam = keras.optimizers.Adam(lr=0.0001)
            model.compile(optimizer=adam, loss='mse')
            
            # #uncomment lines below to the determine number of epochs:
            # history = model.fit(high_x_train_transformed, high_y_train_transformed, epochs = epochs[1], 
            # validation_data=(high_x_test_transformed,high_y_test_transformed))
            # # generate train vs val loss plot
            # processing.plot_train_vs_val_loss(history, 'High_price_loss_graph.png')
        
            model.fit(high_x_train_transformed, high_y_train_transformed, batch_size=1, epochs = epochs[1], verbose = 0, shuffle = False)
            results = model.predict(high_x_test_transformed, batch_size=1)  
            final_results = scaler_high_y.inverse_transform(np.array(results))     
        
            rmse = sqrt(mean_squared_error(high_y_test.iloc[num_steps[1]:],final_results))
            print(f'Test RMSE: {r+1} : {rmse}')
            experiment_results[f'{r}'] = final_results.flatten()
            error_scores.append(rmse)
        
        experiment_results['mean']= experiment_results.mean(axis=1)
        experiment_results.to_csv('RNN_high_predictions.csv', index = False)
        
        df_results = pd.DataFrame()
        df_results['results'] = error_scores
        print(df_results.describe())
        df_results.to_csv('RNN_high_RMSE_experiment_fixed.csv',index = False)


   
        processing.plot_val_vs_actual(experiment_results['mean'], high_y_test, 'Actual High vs Predicted Plot.png' ,'High',num_steps[1])
        
        final_df_results_high = pd.DataFrame(experiment_results['mean'][:-1],index = high_y_test.iloc[num_steps[1]:-1].index,
                                     columns = ['pred_high'])

        final_df_results = pd.concat([final_df_results_high, final_df_results], axis=1, join="inner")
        
        mask3 = df_final.index >= final_df_results.index[0]
        
        final_df_results['actual_low'] = df_final[mask3]['low']
        final_df_results['close'] = df_final[mask3]['close']
        final_df_results['open'] = df_final[mask3]['open']
        final_df_results['actual_high'] = df_final[mask3]['high']
        final_df_results['off_by'] = final_df_results['pred_low']-final_df_results['actual_low']
        final_df_results['return'] = final_df_results['close'] - final_df_results['pred_low']
        final_df_results['return_plow_minus_phigh'] = final_df_results['pred_high'] - final_df_results['pred_low']

        
        #join results with timestamp for high and low daily price csv
        final_df_results = final_df_results.join(timestamp, how = 'left')
        final_df_results.to_excel('RNN_4_20.xlsx')

        value = Overall_Return(final_df_results,return_type = return_type)
        print(f'RNN return over 3 months: {value}')
        # print(f'RNN RMSE over {test_window}: {rmse_low}')



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

    

