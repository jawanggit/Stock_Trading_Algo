
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


import processing as p
import warnings
warnings.filterwarnings("ignore")


class GBR_(object):

    def Daily_Low(self,df_model, alpha_l, test_window = 30):

        a_low = [] #list of actual daily low price 
        p_low = []
        close_lst = []
        dates_lst = []
        open_lst = []   
        
        #alpha_l closer to 0 means model will overpredict low price and you will enter buy order more
        model = GradientBoostingRegressor(n_estimators=500, loss = 'quantile',
                                                max_depth=4, learning_rate=.05, subsample=0.5,
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

        results['return'] = results['close'] - results['pred_low']
        
        #RMSE calculation
        rmse_low = mean_squared_error(results['actual_low'].values,results['pred_low'], squared = False)

        return results, rmse_low

    def Daily_High(self, df_model, results, alpha_h, test_window = 30):
        
        actual_high_lst = []
        pred_high_lst = [] 
        
        #lower alpha means you will have 
        model = GradientBoostingRegressor(n_estimators=500, loss = 'quantile',
                                                max_depth=4, learning_rate=.05, subsample=0.5,
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

        return results

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
            
        #base case
        for index, row in results.iterrows():
            if row['off_by'] >0:
                overall_return += row['return']
        return overall_return-results.iloc[-1,5] # removes most recent actual price which is neg val

class ARIMA_(object):
    def Daily_Low(self,df_model, start_date, test_window = 30):

        index = pd.date_range(start_date, periods=24*test_window,freq='H') # startdate ex: '1/1/21'
        pred_low = []
        
        #convert data to log
        ts_low = df_model['yest_low']
        ts_low = ts_low.apply(math.log)

        #fill in missing data for missing dates 
        
        for i in range(test_window*24):
            low_model = ARIMA(ts_low[:-test_window+i], order=(2,1,0)).fit()
            pred_value = math.exp(low_model.forecast()[0])
            pred_low.append(round(math.exp(pred_value),2))
        
        index = pd.date_range('1/1/2021', periods=test_window*24,freq='H') # startdate ex: '1/1/2021'
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

        return results
        
    def Daily_High(self,df_model,results, test_window = 30):
        
        
        pred_high = []  
        
         #convert data to log
        ts_high = df_model['yest_high']
        ts_high = ts_high.apply(math.log)
        
        for i in range(test_window):
            high_model = ARIMA(ts_high[:-test_window+i], order=(2,1,0)).fit()
            pred_value = high_model.predict(start=len(ts_high)-1)
            pred_high.append(round(math.exp(pred_value),2))
            
        
        results['actual_high'] = df_final['high'].iloc[-test_window:].values
        results['pred_high'] = pred_high
        results['return_phigh_minus_plow'] = results['pred_high'] - results['pred_low']

        return results


    def Overall_Return(self,results, return_type = 'base'):
        overall_return = 0
        index_lst = []

        if return_type == 'optimal':
            for index, row in results.iterrows():
                if row['off_by'] >0:
                    if row['actual_high'] >row['pred_high']:
                        index_lst.append(index)
                        overall_return += row['return_pred_h-pred_l']
                    else:
                        index_lst.append(index)
                        overall_return += row['return']
            return overall_return-results.iloc[-1,6] # removes most recent actual price which is neg val
            
        #base case
        if return_type == 'base':
            for index, row in results.iterrows():
                if row['off_by'] >0:
                    index_lst.append(index)
                    overall_return += row['return']
            return overall_return-results.iloc[-1,6] # removes most recent actual price which is neg val
        
        #pred_low > open (test1)
        if return_type == 'test1':
            for index, row in results.iterrows():
                if row['open'] > row['pred_low'] and row['off_by']>0:
                    index_lst.append(index)
                    overall_return += row['return']
            return overall_return-results.iloc[-1,6] # removes most recent actual price which is neg val



# class RNN_(object):

#     def Daily_Low(self,df_model, test_window = 30):
#     def Daily_High(self,df_model, test_window = 30):
#     def Overall_Return(self,results, return_type = 'base'):

def testing_GBR(alpha_l_list, alpha_h_list,test_window):
    
    data = collections.defaultdict(list)
    for al in alpha_l_list:
        for ah in alpha_h_list:
                GBR_test = GBR_()
                alpha_l = al
                alpha_h = ah
    
                #generate daily low and high price prediction for each day in test window
                df_model_with_low,rmse_low = GBR_test.Daily_Low(df_model, alpha_l, test_window)
                results = GBR_test.Daily_High(df_model,df_model_with_low, alpha_h, test_window)

                #join results with high and low daily price csv
                results['dates']=pd.to_datetime(results['dates'])
                results = results.set_index('dates')
                results = results.join(timestamp, how = 'left')
                
                #Overall return for test window assuming fixed number of shares purchased and sold during each buy and sell transaction
                value = GBR_test.Overall_Return(results,return_type = return_type)
                
                data['RMSE_low'].append(rmse_low)
                data['Return'].append(value)
                data['Alpha_Low'].append(al)
                data['Alpha_High'].append(ah)
                data['Window'].append(60)
                data['Estimators'].append(500)
                data['Learning_Rate'].append(.05)
    
    df = pd.DataFrame.from_dict(data)
    df.to_csv('testing_GBR_60day_v4.csv')







if __name__ == '__main__':

    #df_hourly = p.PullHourlyData('Dal')
    #df = p.PullData('DAL', '2021-04-1')
    df = pd.read_csv('stock_data.csv',index_col = 'date')
    df.index = pd.to_datetime(df.index)




    df_model , df_final = p.AddIndicators(df, '48.54')

    timestamp = pd.read_csv('timestamp_high_low.csv', index_col = 0)
    timestamp.index = pd.to_datetime(timestamp.index)
    
    
    testing_mode = True

    #uncomment to set testing mode
    
  
    #Gradient Boosting
    #parameters
    if testing_mode:

        test_window = 60
        return_type = 'optimal'
        
        alpha_l_list = [.3]
        alpha_h_list = [.9,.8,.7,.6]
        
        testing_GBR(alpha_l_list, alpha_h_list,test_window)
    else:
        
        #test one scenario and generate trade history
        GBR_test = GBR_()
        test_window = 60
        alpha_l = .3
        alpha_h = .95
        return_type = 'optimal'
    
        #generate daily low and high price prediction for each day in test window
        df_model_with_low,rmse_low = GBR_test.Daily_Low(df_model, alpha_l, test_window)
        results = GBR_test.Daily_High(df_model,df_model_with_low, alpha_h, test_window)
        
        #join results with high and low daily price csv
        results['dates']=pd.to_datetime(results['dates'])
        results = results.set_index('dates')
        results = results.join(timestamp, how = 'left')
        results.to_excel('GBR_test_.3_.95_400_.1.xlsx')

        #Overall return for test window assuming fixed number of shares purchased and sold during each buy and sell transaction
        value = GBR_test.Overall_Return(results,return_type = return_type)
        print(f'GBR return over {test_window}: {value}')
        print(f'GBR RMSE over {test_window}: {rmse_low}')

    



    #ARIMA
    # ARIMA_test = ARIMA_()
    # df_model_with_low = ARIMA_test.Daily_Low(df_model)
    # results = ARIMA_test.Daily_High(df_model,df_model_with_low)
    # df_model_with_low.to_excel('ARIMA_test.xlsx')
    # print(f'ARIMA return over 30 periods: {ARIMA_test.Overall_Return(results)}')

    # RNN

