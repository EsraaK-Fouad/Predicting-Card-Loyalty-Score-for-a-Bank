
# =====================================================================================================
'''                                     Libraries ^_^                                                 '''
#========================================================================================================
        
import pandas as pd
import numpy as np



class Read_Data:
    def __init__(self):
        pass

    # =====================================================================================================
    '''                                     Reduce Data Size ^_^                                                 '''
    #========================================================================================================
    # This function reduces data size by optimizing the datatypes of features when possible.
    
    def reduce_mem_usage(self, df, verbose=True):
        dataframe = df
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = dataframe.memory_usage().sum() / 1024**2

        for col in dataframe.columns:
            col_type = dataframe[col].dtypes
            if col_type in numerics:
                c_min = dataframe[col].min()
                c_max = dataframe[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        dataframe[col] = dataframe[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        dataframe[col] = dataframe[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        dataframe[col] = dataframe[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        dataframe[col] = dataframe[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        dataframe[col] = dataframe[col].astype(np.float32)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        dataframe[col] = dataframe[col].astype(np.float32)
                    else:
                        dataframe[col] = dataframe[col].astype(np.float64)

        end_mem = dataframe.memory_usage().sum() / 1024**2
        if verbose:
            print('Memory usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
        return dataframe
    # =====================================================================================================
    '''                                     Read Data ^_^                                                 '''
    #========================================================================================================
    #  Function to load data into pandas and reduce memory usage


    def read_csv(self, path):
        df = self.reduce_mem_usage(pd.read_csv(path))
        return df
            
    


