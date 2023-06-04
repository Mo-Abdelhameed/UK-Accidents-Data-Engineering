from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
from preprocessing import preprocess_data
from extract_features import extract
from dashboard import create_dashboard
from load_to_db import load_to_postgres

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    'start_date': days_ago(2),
    "retries": 1,
}

dag = DAG(
    'UK_Accidents_1981_pipeline',
    default_args=default_args,
    description='UK Accidents 1981 pipeline',
)
with DAG(
    dag_id = 'UK_Accidents_1981_pipeline',
    schedule_interval = '@once',
    default_args = default_args,
    tags = ['UK-pipeline'],
)as dag:
    preprocessing_data= PythonOperator(
        task_id = 'preprocessing',
        python_callable = preprocess_data,
        op_kwargs={
            "path": '/opt/airflow/data/1981_Accidents_UK.csv'
        },
    )
    extract_features= PythonOperator(
        task_id = 'extract',
        python_callable = extract,
        op_kwargs={
            "path": '/opt/airflow/data/preprocessed.csv'
        },
    )
    load_to_postgres_task=PythonOperator(
        task_id = 'load_to_postgres',
        python_callable = load_to_postgres,
        op_kwargs={
            "path": "/opt/airflow/data/integrated_csv.csv",
            "lookup_path": '/opt/airflow/data/lookup.csv'
        },
    )
    create_dashboard_task= PythonOperator(
        task_id = 'create_dashboard_task',
        python_callable = create_dashboard,
        op_kwargs={
            "filename": "/opt/airflow/data/integrated_csv.csv"
        },
    )

    preprocessing_data >> extract_features >> load_to_postgres_task >> create_dashboard_task