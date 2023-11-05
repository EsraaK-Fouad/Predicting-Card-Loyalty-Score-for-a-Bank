import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import missingno as msno 
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from scipy.stats import norm



class Explore_Data:
    
    
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.features_high_corr = 0
        
    #------------------------ Info ^_^ ---------------------------------
    def summary(self):
        print(self.dataframe.info())
        return self.dataframe.describe(include = 'all')
        
    def count_zeros(self, column):
        # Get the count of Zeros in column 
        count = (column == 0).sum()
        return count    
    #------------------------------- Check null ^_^ ------------------------------------
    '''
    This function counts null in each column in the dataframe and calculate the percent of nulls in the column then return the 
    dataframe consist of 2 columns :  one contains count of null values in each column and second contains percent 
    '''
    def null_values(self , plot =True , count_zero_values = True):
        null_val = pd.DataFrame(self.dataframe.isnull().sum())
        null_val.columns = ['null_val']
        null_val['percent_null'] = round(null_val['null_val'] / len(self.dataframe.index), 5) * 100
        null_val = null_val.sort_values('null_val', ascending = False)

        if plot:
            ax = sns.heatmap(self.dataframe.isnull(), cbar=False)
            msno.heatmap(self.dataframe)
            msno.dendrogram(self.dataframe)
            msno.bar(self.dataframe)
            plt.show()

        if count_zero_values:
            null_val['zero_value'] = self.dataframe.apply(self.count_zeros)
            null_val['total_percent'] = round((null_val['null_val'] + null_val['zero_value']) / len(self.dataframe.index), 5) * 100

            
        return null_val
    
     #------------------------------- CHECK CONSTANT FEATURES ^_^  ------------------------------------
    '''
    This function returns the columns that contain one value a cross all samples
    '''
    def constant_columns(self):
        constant_columns = [[col ,self.dataframe[col].value_counts()] for col in self.dataframe.columns if (self.dataframe[col].nunique()) == 1]
        return constant_columns
    
    #-------------------------------- Check the redundant_features ----------------------------------
    '''
    This Function check if there is a high correlation between 2 features .we set a thershold to 0.98. if any 2 features 
    have a correlation larger than 0.95, put them in list .then return correlation matrix and list 
    
    
    '''
    def redundant_features(self):
        #Creating the Correlation matrix
        cor_matrix = self.dataframe.corr().abs()
        #Select the upper triangular
        upper_tri =  pd.DataFrame(cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(bool)))
        #Select the columns which are having absolute correlation greater than 0.98 and making a list of those columns 
        self.features_high_corr = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        if len(self.features_high_corr) == 0 :
            print("There is no redundant features")
            print("*" * 50)
            self.features_high_corr = "empty"
            return  upper_tri 
        else:
            print(self.features_high_corr)
            return  upper_tri
        
    #--------------------------------------- cardinality ^_^ ---------------------------------------   
    '''
    calculate unique values in each column and returns dataframe consists of count and percent. This helps us to find column that have 
    high cardinality 
    '''
    def cardinality(self):
        unique_val = pd.DataFrame(np.array([len(self.dataframe[col].unique()) for col in self.dataframe.columns ]) , index=self.dataframe.columns)
        unique_val.columns = ['unique_val']
        unique_val['percent_'] = round(unique_val['unique_val'] / len(self.dataframe.index), 2) * 100
        unique_val = unique_val.sort_values('percent_', ascending = False)
        return unique_val
    
    #------------------------------- Check duplication ^_^  ------------------------------------
    '''
    This function counts duplicated rows in the dataframe 
    '''
    def duplicated_values(self):
        return print("Number of duplicated rows" , self.dataframe.duplicated().sum())
    
   
    

     #--------------------------------------- drop columns  ^_^ ---------------------------------------   
    '''
    This function drop rows that contain null values.
    '''
    
    def drop_col(self):
        
        self.dataframe = self.dataframe.drop(self.features_high_corr + ['Id'] , axis = 1 , inplace = True ) 
        return self.dataframe
     #---------------------------------------  count of each category   ^_^ ---------------------------------------   
    '''
    This function returns the count of each category in the column as percetage. Also, it put them in a nice plot to imgine the difference between them.  
    '''
    
    def imbalance(self,col):
        print(self.dataframe[col].value_counts(normalize=True))
        self.dataframe[col].value_counts().plot(kind='bar')
        plt.show()
        print()


    def plot_object_features(self, col_excluded):
        
        object_features = self.dataframe.describe(include = 'object').columns
        features_to_plot = list(set(object_features)- set(col_excluded))
        for index in range(len(features_to_plot)):   
            self.imbalance(features_to_plot[index])
            
    def detect_outlier(self , col):
        ''' Detection '''
        # IQR
        # Calculate the upper and lower limits
        Q1 = self.dataframe[col].quantile(0.25)
        Q3 = self.dataframe[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5*IQR
        upper = Q3 + 1.5*IQR

        # Create arrays of Boolean values indicating the outlier rows
        upper_array = np.where(self.dataframe[col]>=upper)[0]
        lower_array = np.where(self.dataframe[col]<=lower)[0]
        print("Number of outliers in this feature is : " , upper_array.shape[0] +lower_array.shape[0])

        return upper_array , lower_array
    
    def visualize_outlier(self):
        plt.figure(figsize=(50,20))
        num_data = self.dataframe.select_dtypes(include=np.number)
        num_data.plot(
        kind='box', 
        subplots=True, 
        sharey=False, 
        figsize=(50, 20) )

        # increase spacing between subplots
        plt.subplots_adjust(wspace=1) 
        plt.show()
        
    def visualize_correlation(self):
        plt.figure(figsize=(20,20))
        num_data = self.dataframe.select_dtypes(include=np.number)
        sns.heatmap(num_data.corr(), annot=True)
        plt.show()
        
    def detect_skewed_features(self):
        num_data = self.dataframe.select_dtypes(include=np.number)
        skewess = num_data.skew().sort_values(ascending =False)
        skewness= pd.DataFrame({'skew':skewess})
        return skewness

    
    
    def normality_test(self , col ):
        k2, p = stats.normaltest(self.dataframe[col])
        alpha = 0.05
        print("p = {:g}".format(p))
        if p < alpha:  
            print("The null hypothesis can be rejected . The data doesn't follow normal distribution")
        else:
            print("The null hypothesis cannot be rejected. The data is normally distributed")
            
        #create Q-Q plot with 45-degree line added to plot
        fig = sm.qqplot(self.dataframe[col], line='45')
        plt.show()
        
        
        # another way to perform graphical method
    def normality_plot(self,col):
        """
        1. Draw distribution plot with normal distribution fitted curve
        2. Draw Quantile-Quantile plot 
        """
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        sns.distplot(self.dataframe[col], fit=norm, ax=axes[0])
        axes[0].set_title('Distribution Plot')
        #`probplot` generates a probability plot, which should not be confused with a Q-Q or a P-P plot.  Statsmodels has more extensive functionality of this type, see ``statsmodels.api.ProbPlot``
        axes[1] = stats.probplot((self.dataframe[col]), plot=plt)
        plt.tight_layout()
        plt.show()
    def Hyposthesis_test(self ,df1 , df2 ,name_col ):
        # Calculate Pearson correlation coefficient
        corr_coef, p_value = stats.pearsonr(df1, df2 )
        # Print the results
        print(f"Pearson correlation coefficient: {corr_coef:.4f}")
        print(f"P-value: {p_value:.4f}")
        if p_value < 0.05 :
            print("Reject null hypothesis and There is a correlation between {} and loyalty score ".format(name_col))
        else:
            print("Fail to reject null hypothesis and There is no correlation between {} and loyalty score ".format(name_col))

        plt.scatter(df1, df2)
        plt.title("plot loyalty score vs {}".format(name_col))
        plt.xlabel(name_col )
        plt.ylabel("loyalty score")
        plt.show()


        
        
            
    


