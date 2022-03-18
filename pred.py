

from airflow.models import DAG

from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.utils.task_group import TaskGroup

from datetime import datetime

from dagtest16nn import readdata
from dagtest16nn import cyclefit
            
from dagtest16nn import cyclepredict
from dagtest16nn import storedata

from utils.load_data import load_data
from utils.preprocess_data import preprocess_data
from utils.experiment import experiment
from utils.track_experiments_info import track_experiments_info
from utils.fit_best_model import fit_best_model
from utils.save_batch_data import save_batch_data


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

    
    # task: 1
    with TaskGroup('creating_storage_structures') as read_data:

        # task: 1.1
       # creating_experiment_tracking_table = PostgresOperator(
        #    task_id="creating_experiment_tracking_table",
        #    postgres_conn_id='postgres_default',
        #    sql='sql/create_experiments.sql'
        #)

        # task: 1.2
        #creating_batch_data_table = PostgresOperator(
       #     task_id="creating_batch_data_table",
       #     postgres_conn_id='postgres_default',
       #     sql='sql/create_batch_data_table.sql'
      #)

    # task: 2
     fetching_data = PythonOperator(
        task_id='fetching_data',
        python_callable=readdata

    )
    
    # task: 3
    with TaskGroup('preparing_data') as train_data:

        # task: 3.1
        traindata = PythonOperator(
            task_id='preprocessing',
            python_callable=cyclefit
        )

      
    # task: 5
    with TaskGroup('after_crossvalidation') as predict:

        # =======
        # task: 5.1        
        saving_results = PythonOperator(
            task_id='saving_results',
            python_callable=cyclepredict
        )
    # task: 5
    with TaskGroup('after_crossvalidation') as saveresult:

        # =======
        # task: 5.1        
        saving_results = PythonOperator(
            task_id='saving_results',
            python_callable=storedata
        )
      

    creating_storage_structures >> fetching_data >> preparing_data >> tuning >> after_crossvalidation