
import pandas as pd
import math
import csv
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
import sklearn.model_selection as cv
import sklearn.datasets as datasets
# from sklearn.model_selection import GridSearchCV
from statsmodels.tsa.arima.model import ARIMA
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers


import processing as p
import warnings
warnings.filterwarnings("ignore")


class GBR_(object):

    def Daily_Low(self,df_model, test_window = 30):

        a_low = [] #list of actual daily low price 
        p_low = []
        close_lst = []
        dates_lst = []
        open_lst = []   

        model = GradientBoostingRegressor(n_estimators=300, loss = 'quantile',
                                                max_depth=4, learning_rate=.1, subsample=0.5,
                                                random_state=154, alpha =.40)

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
        # results['open_minus_pred'] = results['open'] - results['pred_close']
        # results['actual_minus_pred'] = results['actual']-results['pred_close']
        results['off_by'] = results['pred_low']-results['actual_low']

        results['return'] = results['close'] - results['pred_low']

        return results

    def Daily_High(self, df_model, results, test_window = 30):
        
        actual_high_lst = []
        pred_high_lst = [] 

        model = GradientBoostingRegressor(n_estimators=300, loss = 'quantile',
                                                max_depth=4, learning_rate=.1, subsample=0.5,
                                                random_state=154, alpha = .7)

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
        for index, row in results.iterrows():
            if row['off_by'] >0:
                index_lst.append(index)
                overall_return += row['return']
        return overall_return-results.iloc[-1,6] # removes most recent actual price which is neg val

class ARIMA_(object):
    def Daily_Low(self,df_model, test_window = 30):

        pred_low = []
        
        #convert data to log
        ts_low = df_model['yest_low']
        ts_low = ts_low.apply(math.log)

        #fill in missing data for missing dates 
        
        for i in range(test_window):
            low_model = ARIMA(ts_low[:-test_window+i], order=(2,1,0)).fit()
            pred_value = low_model.predict(start=len(ts_low)-1)
            pred_low.append(round(math.exp(pred_value),2))
            
        a_low = df_final['low'].iloc[-test_window:].values
        close_lst = df_final['close'].iloc[-test_window:].values
        open_lst = df_final['open'].iloc[-test_window:].values
        dates = df_final.index[-test_window:]
        
            
        d = {'dates':dates,'actual_low': a_low, 'pred_low':pred_low, 'close': close_lst, 'open':open_lst}

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


if __name__ == '__main__':

    df = p.PullData('DAL', '2021-03-31')
    df_model , df_final = p.AddIndicators(df, '48.80')

    #Gradient Boosting
    test_window = 60
    return_type = 'test1'
    GBR_test = GBR_()
    df_model_with_low = GBR_test.Daily_Low(df_model, test_window)
    results = GBR_test.Daily_High(df_model,df_model_with_low, test_window)
    results.to_excel('GBR_test.xlsx')

    print(f'GBR return over {test_window}: {GBR_test.Overall_Return(results,return_type ==return_type)}')

       # #ARIMA
    # ARIMA_test = ARIMA_()
    # df_model_with_low = ARIMA_test.Daily_Low(df_model)
    # results = ARIMA_test.Daily_High(df_model,df_model_with_low)
    # df_model_with_low.to_excel('ARIMA_test.xlsx')
    # print(f'ARIMA return over 30 periods: {ARIMA_test.Overall_Return(results)}')

    # RNN

