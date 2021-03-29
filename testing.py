
import pandas as pd
import csv
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
import sklearn.model_selection as cv
import sklearn.datasets as datasets
from sklearn.model_selection import GridSearchCV
import processing as p

class GBR_(object):

    def Daily_Low(self,df_model,df_final, test_window = 30):

        a_low = [] #list of actual daily low price 
        p_low = []
        close_lst = []
        dates_lst = []   

        model = GradientBoostingRegressor(n_estimators=300, loss = 'quantile',
                                                max_depth=4, learning_rate=.1, subsample=0.5,
                                                random_state=154, alpha =.7)

        for i in range(test_window):
            X_train = df_model.iloc[:-test_window-i]
            X_test = df_model.iloc[-test_window+i:]
            y_train = df_final['low'].iloc[:-test_window-i].values
            y_test = df_final['low'].iloc[-test_window+i:].values
                           
            model.fit(X_train, y_train)
            
            a_low.append(y_test[:1])
            p_low.append(model.predict(X_test[:1]))
            close_lst.append(df_final['close'].iloc[-30+i:].values[:1])
            dates_lst.append(X_test.index[:1])

            actual_low = np.array(a_low).flatten()
            pred_low = np.array(p_low).flatten()
            close = np.array(close_lst).flatten()
            dates = np.array(dates_lst).flatten()

        d = {'dates':dates,'actual_low': actual_low, 'pred_low':pred_low, 'close': close}

        results = pd.DataFrame(d)
        # results['open_minus_pred'] = results['open'] - results['pred_close']
        # results['actual_minus_pred'] = results['actual']-results['pred_close']
        results['off_by'] = results['pred_low']-results['actual_low']

        results['return'] = results['close'] - results['pred_low']

        return results

    def Daily_High(self, df_model, df_final,results, test_window = 30):
        
        actual_high_lst = []
        pred_high_lst = [] 
        
        for i in range(30):
            X_train = df_model.iloc[:-30-i]
            X_test = df_model.iloc[-30+i:]
            y_train = df_final['high'].iloc[:-30-i].values
            y_test = df_final['high'].iloc[-30+i:].values
            
            model = GradientBoostingRegressor(n_estimators=300, loss = 'quantile',
                                                max_depth=4, learning_rate=.1, subsample=0.5,
                                                random_state=154, alpha = .6)
                
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
            return overall_return-results.iloc[-1,5] # removes most recent actual price which is neg val
            
        #base case
        for index, row in results.iterrows():
            if row['off_by'] >0:
                index_lst.append(index)
                overall_return += row['return']
        return overall_return-results.iloc[-1,5] # removes most recent actual price which is neg val

        
        
if __name__ == '__main__':

    df = p.PullData('DAL', '2021-03-29')
    df_model , df_final = p.AddIndicators(df, '47.44')

    #Gradient Boosting
    GBR_test = GBR_()
    df_model_with_low = GBR_test.Daily_Low(df_model, df_final)
    results = GBR_test.Daily_High(df_model, df_final,df_model_with_low)
    print(GBR_test.Overall_Return(results))

    #Arima
    
