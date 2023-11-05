
#-----------------------------------------------------------------

#                           Import Libraries ^_^ 

#--------------------------------------------------------------------

import joblib
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn import ensemble
import numpy as np
import matplotlib.pyplot as plt



class Model:
    
    
    def __init__(self , data ): 
        self.data =  data 
      
    
   #---------------------------------------- Aggregation function -----------------------------
    '''
    This Function takes dataframe and aggregation function and applies this aggregation function on this dataframe .Then rename it and return new dataframe
    '''       
    def aggregation(self , agg_func):
        agg_trans = self.data.groupby(['card_id']).agg(agg_func)
        agg_trans.columns = [ '_'.join(col).strip() for col in agg_trans.columns.values]
        agg_trans.reset_index(inplace=True)
        return  agg_trans
       
        
     #---------------------------------------- Split function -----------------------------
    '''
    This Function splits data into train and test datasets with ratio 80:20 respectively.Then returns splited data
    ''' 
    def split_data(self , df):
        X =  df.drop(['score_min' , 'card_id'], axis = 1 )
        y =  df['score_min']
        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
        return X_train , X_test,  y_train, y_test
    
    
    #---------------------------------------- Build model function -----------------------------
    '''
    This Function builds our model.Using this function, we can build 2 models (optionally).
    
    '''     
    def build_model(self ,  model_name):
        if model_name == 'lgb':
            model = lgb.LGBMRegressor( n_estimators = 20000, nthread = 4, n_jobs = -1)
        else:
            params = {
                        "n_estimators": 500,
                        "max_depth": 4,
                        "min_samples_split": 5,
                        "learning_rate": 0.01,
                        "loss": "squared_error",
                    }
            model = ensemble.GradientBoostingRegressor( **params )         
        return model
        
        
        
    def train_model(self, model_name,X_train ,  y_train ,X_test ,y_test):
        model = self.build_model(model_name)
        if model_name ==  'lgb':
            model.fit(X_train, y_train, 
            eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric='rmse',
            verbose=1000, early_stopping_rounds=200)
        else:
            model.fit(X_train , y_train)
        return model
    
    
    
    def evaluation(self , model, model_name, X_train , X_test,  y_train, y_test):     
        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        #squared True returns MSE value, False returns RMSE value.
        mae = mean_absolute_error(y_true= y_test ,y_pred=y_pred)
        mse = mean_squared_error(y_true=y_test,y_pred=y_pred) #default=True
        rmse = mean_squared_error(y_true=y_test,y_pred=y_pred , squared= False)
        r2 = model.score(X_test , y_test)
      #squared True returns MSE value, False returns RMSE value.
        mae_train = mean_absolute_error(y_true= y_train ,y_pred=y_pred_train)
        mse_train = mean_squared_error(y_true=y_train,y_pred=y_pred_train) #default=True
        rmse_train = mean_squared_error(y_true=y_train,y_pred=y_pred_train , squared= False)
        r2_train = model.score(X_train , y_pred_train)
        print("MAE in test data :",mae)
        print("MAE in train data :",mae_train)

        print("MSE in test data:",mse)
        print("MSE in train data:",mse_train)

        print("RMSE in test data:",rmse)
        print("RMSE in train data:",rmse_train)

        print("R square in test data:",r2)
        print("R square in train data:",r2_train)
        plt.scatter(y_test, y_pred, color='red', label='Predicted Score')
        plt.scatter(y_test, y_test, color='blue', label='Real Score')
        plt.xlabel('Real Score')
        plt.ylabel('Predicted Score')
        plt.title('Real vs Predicted Scores in test data ')
        plt.show()
        plt.scatter(y_train, y_pred_train, color='black', label='Predicted Score')
        plt.scatter(y_train, y_train, color='red', label='Real Score')
        plt.xlabel('Real Score')
        plt.ylabel('Predicted Score')
        plt.title('Real vs Predicted Scores in train data')
        plt.show()
        
        
    def save_model(self , model , name):
        joblib.dump(model , "model_{}.pkl".format(name))
       
    
    
    def feature_importance(self , model ,X_train,X_test, y_test):
        
        feature_importance = model.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + 0.5
        fig = plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.barh(pos, feature_importance[sorted_idx], align="center")
        plt.yticks(pos, np.array(X_train.columns)[sorted_idx])
        plt.title("Feature Importance (MDI)")

        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
        sorted_idx = result.importances_mean.argsort()
        plt.subplot(1, 2, 2)
        plt.boxplot(
            result.importances[sorted_idx].T,
            vert=False,
            labels=np.array(X_train.columns)[sorted_idx], 
                    )
        plt.title("Permutation Importance (test set)")
        fig.tight_layout()
        plt.show()



        
            
    


