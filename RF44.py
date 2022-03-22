
# Load all necessary libraries
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm().pandas()



import sys, getopt
import torch

import numpy as np
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.cluster import OPTICS

# forecast monthly births with xgboost
from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from matplotlib import pyplot
import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.cluster import OPTICS
import numpy as np; 
# Data manipulation
# ==============================================================================
import numpy as np
import pandas as pd

# Plots
# ==============================================================================
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
plt.style.use('fivethirtyeight')

# Modelado y Forecasting
# ==============================================================================
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregMultiOutput import ForecasterAutoregMultiOutput
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster


import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.cluster import OPTICS
import pickle
from array import *

ar = []
pred = []

end_validation = 900


forecaster = ForecasterAutoregMultiOutput(
                regressor = RandomForestRegressor(max_depth=29),
                steps     = 24, #60
                lags      = 20 # This value will be replaced in the grid search
             )



def read_data():
    global datasetttt
    dataset4 = pd.read_csv('timedddd.csv',sep = ";") # turned into parallel hexagon array
#dataset4 =  dataset.set_index('time')
    dataset4 =  dataset4.set_index('time')
    datasetttt=dataset4
    
    return datasetttt
    
def fitdata(datasetttt):
    if torch.cuda.is_available():
        dev = "cuda:0"
        
    else:
        dev = "cpu"
        device = torch.device(dev)
        torch.cuda.set_device(-1)
    
    for index in range(datasetttt.shape[1]):
        columnSeriesObj = datasetttt.iloc[:, index]
        columnSeriesObj = pd.Series(list(columnSeriesObj))
        forecaster = ForecasterAutoregMultiOutput(
                regressor = RandomForestRegressor(max_depth=29),
                steps     = 24, #60
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
        lags_grid = [[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24], [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]]
        results_grid = grid_search_forecaster(
                        forecaster  = forecaster,
                        y           = pd.Series(list(columnSeriesObj2)),#data[:,index],table11 = pd.Series(list(table1))   
                        exog        = columnSeriesObj1,#data[:,index],
                        param_grid  = param_grid,
                        lags_grid   = lags_grid,
                        steps       = 24,
                        metric      = 'mean_absolute_error',
                        refit       = False,
                        initial_train_size = 900,
                        return_best = True,
                        verbose     = False
                  )
        return forecaster
        
def predictions(forecaster):
    for index in range(datasetttt.shape[1]):
        columnSeriesObj = datasetttt.iloc[:, index]
        columnSeriesObj = pd.Series(list(columnSeriesObj))
        
        columnSeriesObj1 = columnSeriesObj[48:1024]
        columnSeriesObj2 = columnSeriesObj[24:1000]
    #columnSeriesObj3 = columnSeriesObj[24:1000]
        columnSeriesObj3 = columnSeriesObj[0:976]
        
        forecaster = ForecasterAutoregMultiOutput(
                regressor = RandomForestRegressor(max_depth=14),
                steps     = 24, #60
                lags      = 20 # This value will be replaced in the grid search
             )
        metric, predictions = backtesting_forecaster(
                            forecaster = forecaster,
                            y          = pd.Series(list(columnSeriesObj3)),
                            exog        = columnSeriesObj2,
                            initial_train_size = len(columnSeriesObj[:end_validation]),
                            steps      = 24,
                            metric     = 'mean_absolute_error',
                            refit      = False,
                            verbose    = False)
#for index in range(data.shape[1]):
    #columnSeriesObj = data.iloc[:, index]
    #columnSeriesObj = pd.Series(list(columnSeriesObj))                      
        fig, ax = plt.subplots(figsize=(12, 8))
        columnSeriesObj.iloc[predictions.index].plot(linewidth=2, label='real', ax=ax)
        predictions.plot(linewidth=2, label='prediction', ax=ax)
        ax.set_title('Prediction vs real orders')
        ax.legend();
        ar.append(predictions.copy())
        print(ar)
        
def save_data(forecaster):
    forecaster = fitdata(datasetttt)
    PIK = "models.pckl"
    pred.append(forecaster)
    with open("models.pckl", "wb") as f:
        for forecaster in pred:
             pickle.dump(forecaster, f)
            
    with open(PIK, "rb") as f:
        print(pickle.load(f))

def main(argv):       
    datasetttt = read_data()     
              
    predictions(forecaster)
    save_data(forecaster)
    #d[index, :] = d[index, :] + predictions

if __name__ == "__main__":
    """
    Program entry point.
    """
    main(sys.argv[1:])
