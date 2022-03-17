import numpy as np; 
# Data manipulation
# ==============================================================================
import numpy as np
import pandas as pd
#import matplotlib
import dagtest4nn
from dagtest4nn import cyclefitpredict

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
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
#from skforecast.ForecasterAutoreg import ForecasterAutoreg
import sys
sys.path.append( '/opt/aws-mwaa-local-runner/dags/skforecast/skforecast')
from skforecast.ForecasterAutoregMultiOutput import ForecasterAutoregMultiOutput
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





from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
from datetime import timedelta
# The DAG object; we'll need this to instantiate a DAG

# Operators; we need this to operate!
from airflow.operators.bash_operator import BashOperator
from airflow.utils.dates import days_ago

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
# Importing mlflow and its sklearn component
import mlflow
import mlflow.sklearn





ar = []
end_validation = 900
dataset4 = pd.read_csv('/opt/dags/timedddd.csv',sep = ";")
datasetttt = dataset4

def create_dag(dag_id,
               schedule,
               dag_number,
               default_args):

    #def hello_world_py(*args):
    #    print('Hello World')
    #    print('This is DAG: {}'.format(str(dag_number)))

    dag = DAG(dag_id,
              schedule_interval=schedule,
              default_args=default_args)

    with dag:
        t1 = PythonOperator(
            task_id='cyclefitpredict',
            python_callable=cyclefit)
        #t2 = PythonOperator(
         #   task_id='cyclefitpredict',
         #   python_callable=cyclepredict)
        #t1 >> t2
    return dag


# build a dag for each number in range(10)

for index in range(datasetttt.shape[1]):
#for n in range(1, 4):
    dag_id = 'cyclefitpredict_{}'.format(str(datasetttt.shape[1]))

    default_args = {'owner': 'cyclefitpredict',
    'depends_on_past': False,
    'start_date': days_ago(31),
    'retries': 1,
    'retry_delay': timedelta(minutes=2)
                    }

    schedule = '@daily'
    dag_number = datasetttt.shape[1]

    globals()[dag_id] = create_dag(dag_id,
                                  schedule,
                                  dag_number,
                                  default_args)
