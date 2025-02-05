import os
import requests
from airflow.decorators import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook
#from airflow.operators.postgres_operator import PostgresOperator
from airflow.utils.log.logging_mixin import LoggingMixin
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
    schedule="0 0 * * *", # running daily at midnight
    start_date=pendulum.datetime(2025, 1, 1, tz='UTC'),
    catchup=False,
    dagrun_timeout=datetime.timedelta(minutes=60),
    default_args=default_args,
    tags=['ml', 'churn']
)

@task
def get_data():
    postgres_hook = PostgresHook(postgres_conn_id='postgres_churn')
    query = "SELECT * FROM customers;"
    df = postgres_hook.get_pandas_df(query)
    print(f"Data shape: {df.shape}")
    print(f"Data Head: {df.head()}")
    print(f"Data info: {df.info()}")
    logger = LoggingMixin().log
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"\nData Head: \n{df.head()}")
    return df 


@task
def preprocessing(df):
   df = df.copy()
   
   features = ['age', 'contract_length', 'monthly_charges', 'contract_type']
   target = 'churn'
   
   X = df[features]
   y = df[target]
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
   numerical_features = ['age', 'contract_length', 'monthly_charges']
   categorical_features = ['contract_type']

    # pipeline for numerical features, imputing null values and scaling
   numeric_transformer = Pipeline(steps=[
       ('imputer', SimpleImputer(strategy='median')),
       ('scaler', StandardScaler())
   ])
    # pipeline for categorical features, imputing null values and one hot encoding contract type
   categorical_transformer = Pipeline(steps=[
       ('imputer', SimpleImputer(strategy='most_frequent')),
       ('onehot', OneHotEncoder(drop='first', sparse_output=False))
   ])

   preprocessor = ColumnTransformer(
       transformers=[
           ('num', numeric_transformer, numerical_features),
           ('cat', categorical_transformer, categorical_features)
       ])
   

   X_train_processed = preprocessor.fit_transform(X_train)
   
   # calling .transform instead of .fit_transform on the test set to prevent data leakage
   X_test_processed = preprocessor.transform(X_test)

    # using SMOTE for class imbalance
   smote = SMOTE(random_state=42)
   X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train)

   numeric_features = numerical_features
   categorical_features_encoded = [f"contract_type_{cat}" for cat in 
                                 preprocessor.named_transformers_['cat']
                                 .named_steps['onehot']
                                 .get_feature_names_out(['contract_type'])]
   feature_names = numeric_features + list(categorical_features_encoded)

   X_train_processed = pd.DataFrame(X_train_balanced, columns=feature_names)
   X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names)

   return X_train_processed, X_test_processed, y_train_balanced, y_test, preprocessor


@task 
def train_model(X_train, y_train):
    logistic_reg = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    )

    logistic_reg.fit(X_train, y_train)
    return logistic_reg

@task
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test,y_pred)

    class_report = classification_report(y_test, y_pred, output_dict=True)

    # logging the results
    logger = LoggingMixin().log
    logger.info(f"Model Accuracy: {accuracy}")
    logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
    return {
        'accuracy': accuracy,
        'classification_report': class_report
    }

@task
def save_model(model,preprocessor):
    # making sure the directory exists
    os.makedirs('models', exist_ok=True)
    model_path = 'models/logistic_regression_churn_model.pkl'
    preprocessor_path = 'models/churn_preprocessor.pkl'

    with open(model_path, 'wb') as f:
        pickle.dump(model,f)

    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)

    logger = LoggingMixin().log
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Preprocessor saved to {preprocessor_path}")
    
    return model_path, preprocessor_path


@task
def model_workflow(df):
    X_train, X_test, y_train, y_test, preprocessor = preprocessing(df)
    print(X_train.haed())
    print(X_test.head())
    
    trained_model = train_model(X_train, y_train)
    print(trained_model)
    # Evaluate the model
    evaluation_results = evaluate_model(trained_model, X_test, y_test)
    
    # Save the model and preprocessor
    model_path, preprocessor_path = save_model(trained_model, preprocessor)
    
    return {
        'evaluation_results': evaluation_results,
        'model_path': model_path,
        'preprocessor_path': preprocessor_path
    }

customer_data = get_data()
workflow_results = model_workflow(customer_data)
