import numpy as np; 
# Data manipulation
# ==============================================================================
import numpy as np
import pandas as pd
from array import *
# Plots
# ==============================================================================
#import matplotlib.pyplot as plt
#import seaborn as sns
#%matplotlib inline
#from statsmodels.graphics.tsaplots import plot_acf
#from statsmodels.graphics.tsaplots import plot_pacf
#plt.style.use('fivethirtyeight')

# Modelado y Forecasting
# ==============================================================================
#import sklearn
#from sklearn.linear_model import Ridge
#from lightgbm import LGBMRegressor
#from sklearn.pipeline import make_pipeline
#from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import mean_absolute_error
#from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregMultiOutput import ForecasterAutoregMultiOutput
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster

# Warnings configuration
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

# forecast monthly births with xgboost
from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from matplotlib import pyplot

ar = []
end_validation = 900
dataset4 = pd.read_csv('timedddd.csv',sep = ";")
datasetttt = dataset4
for index in range(datasetttt.shape[1]):
        columnSeriesObj = datasetttt.iloc[:, index]

def readdata():
    dataset4 = pd.read_csv('timedddd.csv',sep = ";") # turned into parallel hexagon array
#dataset4 =  dataset.set_index('time')
    dataset4 =  dataset4.set_index('time')
    dataset4.head()
    datasetttt = dataset4
    
    ar = []
#data = data.astype(int)
    d = datasetttt
#d = 0
    d = np.asarray(d)
    end_validation = 900
    return dataset4,datasetttt
    
   
def cyclefitpredict():    
    for index in range(datasetttt.shape[1]):
        columnSeriesObj = datasetttt.iloc[:, index]
        columnSeriesObj = pd.Series(list(columnSeriesObj))
        forecaster = ForecasterAutoreg(
                regressor = XGBRegressor(objective='reg:squarederror', n_estimators=1000),
                #steps     = 24, #60
                lags      = 20 # This value will be replaced in the grid search
             ) 
        columnSeriesObj1 = columnSeriesObj[48:1024]
        columnSeriesObj2 = columnSeriesObj[24:1000]
    #columnSeriesObj3 = columnSeriesObj[24:1000]
        columnSeriesObj3 = columnSeriesObj[0:976]
# Regressor's hyperparameters
        param_grid = {'n_estimators': [100, 500],
              'max_depth': [4, 6]}
# Lags used as predictors
        lags_grid = [[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24], 
         [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]]
        
        forecaster.fit(
                    y    = pd.Series(list(columnSeriesObj2)),
                    exog = columnSeriesObj1
                  )
    #return forecaster
                  
#def cyclepredict():  
    #for index in range(datasetttt.shape[1]):
        columnSeriesObj = datasetttt.iloc[:, index]
        metric, predictions = backtesting_forecaster(
                            forecaster = forecaster,
                            y          = pd.Series(list(columnSeriesObj2)),
                            exog       = columnSeriesObj3,
                            initial_train_size = len(columnSeriesObj[:end_validation]),
                            steps      = 24,
                            metric     = 'mean_absolute_error',
                            refit      = False,
                            verbose    = False)
                            
                          

#for index in range(data.shape[1]):
    #columnSeriesObj = data.iloc[:, index]
    #columnSeriesObj = pd.Series(list(columnSeriesObj)) 
#def storedata(predictions): 
   # for index in range(datasetttt.shape[1]):
       # columnSeriesObj = datasetttt.iloc[:, index]  
        #restore                  
        #fig, ax = plt.subplots(figsize=(12, 8))
        #columnSeriesObj.iloc[predictions.index].plot(linewidth=2, label='real', ax=ax)
        #predictions.plot(linewidth=2, label='prediction', ax=ax)
        ax.set_title('Prediction vs real orders')
        ax.legend()
        ar.append(predictions.copy())
        print(ar)
        return predictions
#dataset4 = pd.read_csv('timedddd.csv',sep = ";") # turned into parallel hexagon array
#dataset4 =  dataset.set_index('time')
#dataset4 =  dataset4.set_index('time')
#dataset4.head()
#datasetttt = dataset4
#print(cyclefitpredict())




    #d[index, :] = d[index, :] + predictions

