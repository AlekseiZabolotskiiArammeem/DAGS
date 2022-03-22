
from airflow.models import DAG

from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.utils.task_group import TaskGroup

from datetime import datetime

from RandomForestFull import fitdata
from RandomForestFull import predictions
            
from RandomForestFull import save_data
#from XGBoost4 import storedata



default_args= {
    'owner': 'Alex',
    'email_on_failure': False,
    'email': ['alex@mail.com'],
    'start_date': datetime(2022, 12, 1)
}

with DAG(
    "ml_pipeline",
    description='ML pipeline example',
    schedule_interval='@daily',
    default_args=default_args, 
    catchup=False) as dag:

    
      # task: 3
    with TaskGroup('readdata') as readdata:

        # task: 3.1
        traindata = PythonOperator(
            task_id='learn',
            python_callable=read_data
        )


    # task: 3
    with TaskGroup('fit_data') as fit_data:

        # task: 3.1
        traindata = PythonOperator(
            task_id='learn',
            python_callable=fitdata
        )

      
    # task: 5
    with TaskGroup('prediction') as prediction:

        # =======
        # task: 5.1        
        saving_results = PythonOperator(
            task_id='predict',
            python_callable=predictions
        )
    # task: 5
    with TaskGroup('save') as saveresult:

        # =======
        # task: 5.1        
        saving_results = PythonOperator(
            task_id='saving_results',
            python_callable=save_data
        )
      

    readdata >> fit_data >> prediction >> saveresult