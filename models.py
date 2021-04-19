
import pandas as pd
import math
import csv
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
import sklearn.model_selection as cv
import sklearn.datasets as datasets
# from sklearn.model_selection import GridSearchCV
import collections

from statsmodels.tsa.arima.model import ARIMA 
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
import processing
import warnings
warnings.filterwarnings("ignore")


def Daily_Low(self,df_model, alpha_l, test_window = 30):

    if GBR:

        a_low = [] #list of actual daily low price 
        p_low = []
        close_lst = []
        dates_lst = []
        open_lst = []   
        
        #alpha_l closer to 0 means model will overpredict low price and you will enter buy order more
        model = GradientBoostingRegressor(n_estimators=500, loss = 'quantile',
                                                max_depth=8, learning_rate=.05, subsample=0.5,
                                                random_state=154, alpha = alpha_l)

        for i in range(test_window):
            X_train = df_model.iloc[:-test_window-i]
            X_test = df_model.iloc[-test_window+i:]
            y_train = df_final['low'].iloc[:-test_window-i].values
            y_test = df_final['low'].iloc[-test_window+i:].values
                            
            model.fit(X_train, y_train)
            
            a_low.append(y_test[:1])
            p_low.append(model.predict(X_test[:1]))
            close_lst.append(df_final['close'].iloc[-test_window+i:].values[:1])
            dates_lst.append(X_test.index[:1])
            open_lst.append(df_final['open'].iloc[-test_window+i:].values[:1])

            actual_low = np.array(a_low).flatten()
            pred_low = np.array(p_low).flatten()
            close = np.array(close_lst).flatten()
            open_p = np.array(open_lst).flatten()
            dates = np.array(dates_lst).flatten()

        d = {'dates':dates,'actual_low': actual_low, 'pred_low':pred_low, 'close': close, 'open':open_p}

        results = pd.DataFrame(d)
        results['off_by'] = results['pred_low']-results['actual_low']   
        
        #RMSE calculation
        rmse_low = mean_squared_error(results['actual_low'].values,results['pred_low'], squared = False)

return results, rmse_low



def Daily_High(self, df_model, results, alpha_h, test_window = 30):
    
    if GBR:

        actual_high_lst = []
        pred_high_lst = [] 
        
        #lower alpha means you will have 
        model = GradientBoostingRegressor(n_estimators=500, loss = 'quantile',
                                                max_depth=8, learning_rate=.05, subsample=0.5,
                                                random_state=154, alpha = alpha_h)

        for i in range(test_window):
            X_train = df_model.iloc[:-test_window-i]
            X_test = df_model.iloc[-test_window+i:]
            y_train = df_final['high'].iloc[:-test_window-i].values
            y_test = df_final['high'].iloc[-test_window+i:].values        
                
            model.fit(X_train, y_train)
            
            actual_high_lst.append(y_test[:1])
            pred_high_lst.append(model.predict(X_test[:1]))

            actual_high = np.array(actual_high_lst).flatten()
            pred_high = np.array(pred_high_lst).flatten()
        
        results['actual_high'] = actual_high
        results['pred_high'] = pred_high
        results['return_plow_minus_phigh'] = results['pred_high'] - results['pred_low']

        #RMSE calculation
        rmse_high = mean_squared_error(results['actual_high'].values,results['pred_high'], squared = False)

        return results , rmse_high

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


        # results['return'] = results['close'] - results['pred_low']

        
        return overall_return-results.iloc[-1,5] # removes most recent actual price which is neg val


'''class ARIMA_(object):
    def Daily_Low(self,df_hourly_low, start_date,test_window = 60):

        index = pd.date_range(start_date, periods=24*test_window,freq='H') # startdate ex: '1/1/21'
        pred_low = []
        
        #convert data to log
        # ts_low = pd.Series(df_hourly_low['low'].values,index = df_hourly_low.index)
        ts_low = df_hourly_low
        test_window_in_hours = test_window*24
      
        for i in range(test_window_in_hours):
            low_model = ARIMA(ts_low.iloc[:-test_window_in_hours+i], order=(2,1,0)).fit()
            pred_value = math.exp(low_model.forecast()[0])
            pred_low.append(round(pred_value,2))
        
        pred_low = pd.Series(pred_low, index=index)
        pred_low = pred_low.resample('D').mean()
        
        a_low = df_final['low'].iloc[-test_window:].values
        close_lst = df_final['close'].iloc[-test_window:].values
        open_lst = df_final['open'].iloc[-test_window:].values
        dates = df_final.index[-test_window:]
        
            
        d = {'dates':dates,'actual_low': a_low, 'pred_low':pred_low.values, 'close': close_lst, 'open':open_lst}

        results = pd.DataFrame(d)
        # results['open_minus_pred'] = results['open'] - results['pred_close']
        # results['actual_minus_pred'] = results['actual']-results['pred_close']
        results['off_by'] = results['pred_low']-results['actual_low']

        results['return'] = results['close'] - results['pred_low']

        #RMSE calculation
        rmse_low = mean_squared_error(results['actual_low'].values,results['pred_low'], squared = False)

        return results,rmse_low
    
    
    def Daily_High(self,df_hourly_high,results,start_date, test_window = 60):
        
        index = pd.date_range(start_date, periods=24*test_window,freq='H') # startdate ex: '1/1/21'
        pred_high = []  
        
         #convert data to log
        ts_high = df_hourly_high
        test_window_in_hours = test_window*24
        
        for i in range(test_window_in_hours):
            high_model = ARIMA(ts_high.iloc[:-test_window_in_hours+i], order=(2,1,0)).fit()
            pred_value = math.exp(high_model.forecast()[0])
            pred_high.append(round(pred_value,2))
        
        pred_high = pd.Series(pred_high, index=index)
        pred_high = pred_high.resample('D').mean()
            
        
        results['actual_high'] = df_final['high'].iloc[-test_window:].values
        results['pred_high'] = pred_high.values
        results['return_phigh_minus_plow'] = results['pred_high'] - results['pred_low']

        #RMSE calculation
        rmse_high = mean_squared_error(results['actual_high'].values,results['pred_high'], squared = False)

        return results, rmse_high


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
            trades.to_excel('TRADES_ARIMA_.3_.6_60_500.xlsx')
            return overall_return-results.iloc[-1,5] # removes most recent actual price which is neg val
            
        #base case
        for index, row in results.iterrows():
            if row['off_by'] >0:
                overall_return += row['return']
        return overall_return-results.iloc[-1,5] # removes most recent actual price which is neg val
'''

def testing_GBR(alpha_l_list, alpha_h_list,test_window):
    
    data = collections.defaultdict(list)
    for al in alpha_l_list:
        for ah in alpha_h_list:
                GBR_test = GBR_()
                alpha_l = al
                alpha_h = ah
    
                #generate daily low and high price prediction for each day in test window
                df_model_with_low,rmse_low = GBR_test.Daily_Low(df_model, alpha_l, test_window)
                results,rmse_high = GBR_test.Daily_High(df_model,df_model_with_low, alpha_h, test_window)

                #join results with high and low daily price csv
                results['dates']=pd.to_datetime(results['dates'])
                results = results.set_index('dates')
                results = results.join(timestamp, how = 'left')
                
                #Overall return for test window assuming fixed number of shares purchased and sold during each buy and sell transaction
                value = GBR_test.Overall_Return(results,return_type = return_type)
                
                data['RMSE_low'].append(rmse_low)
                data['RMSE_high'].append(rmse_high)
                data['Return'].append(value)
                data['Alpha_Low'].append(al)
                data['Alpha_High'].append(ah)
                data['Window'].append(60)
                data['Estimators'].append(500)
                data['Learning_Rate'].append(.05)
    
    df = pd.DataFrame.from_dict(data)
    df.to_csv('testing_GBR_60day_v5.csv')


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
    GBR = True
    ARIMA = False
    test_window = 3
    return_type = 'optimal'
    
    #alpha_l_list = [.3]
    #alpha_h_list = [.9,.8,.7,.6]

    #RNN

    
    


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
    if ARIMA_model:
        
        ARIMA_test = ARIMA_()
        df_low,rmse_low = ARIMA_test.Daily_Low(df_hourly_low,'1/1/2021', test_window)
        df_low.to_excel('ARIMA_low.xlsx')
        results = ARIMA_test.Daily_High(df_hourly_high,df_low,'1/1/2021',test_window)
        results.to_excel('ARIMA_full_results.xlsx')

        print(f'ARIMA return over 60 trading days: {ARIMA_test.Overall_Return(results)}')
        print(f'ARIMA RMSE: {rmse_low}')

