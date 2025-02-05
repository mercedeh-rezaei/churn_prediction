import os
import requests
from airflow.decorators import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.postgres.operators.postgres import PostgresOperator
import datetime
import pendulum

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import pickle

default_args = {
    'retries':3,
    'retry_delay': datetime.timedelta(minutes=5),
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
}


@dag(
    dag_id="churn_prediction",
    schedule_interval="0 0 * * *", # running daily at midnight
    start_date=pendulum.datetime(2025, 1, 1, tz='UTC'),
    catchup=False,
    dagrun_timeout=datetime.timedelta(minutes=60),
    default_args=default_args,
    tags=['ml', 'churn']
)
def churn_prediction_pipeline():
    pass

churn_prediction_dag = churn_prediction_pipeline()