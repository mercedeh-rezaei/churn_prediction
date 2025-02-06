import os
import requests
from airflow.decorators import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook
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
import logging

# configuring logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

default_args = {
    'retries': 3,
    'retry_delay': datetime.timedelta(minutes=5),
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
}

@dag(
    dag_id="churn_prediction",
    schedule="0 0 * * *",  # running daily at midnight
    start_date=pendulum.datetime(2025, 1, 1, tz='UTC'),
    catchup=False,
    dagrun_timeout=datetime.timedelta(minutes=60),
    default_args=default_args,
    tags=['ml', 'churn']
)
def model_workflow():
    @task
    def setup_directories():
        """Create necessary directories for storing files"""
        base_dir = '/opt/airflow'
        data_dir = os.path.join(base_dir, 'data')
        temp_dir = os.path.join(data_dir, 'temp')
        model_dir = os.path.join(base_dir, 'models')
        
        for directory in [data_dir, temp_dir, model_dir]:
            os.makedirs(directory, exist_ok=True)
            
        logger = LoggingMixin().log
        logger.info(f"Base directory: {base_dir}")
        logger.info(f"Data directory: {data_dir}")
        logger.info(f"Temp directory: {temp_dir}")
        logger.info(f"Model directory: {model_dir}")
            
        return {
            'data_dir': data_dir,
            'temp_dir': temp_dir,
            'model_dir': model_dir
        }

    @task
    def get_data():
        """Retrieves customer data from PostgresSQL database"""

        logger = LoggingMixin().log
        try:
            # fetching data
            postgres_hook = PostgresHook(postgres_conn_id='postgres_churn')
            query = "SELECT * FROM customers;"
            df = postgres_hook.get_pandas_df(query)
            
            logger.info("Data retrieval successful")
            logger.info(f"Data size: {len(df)}")
            logger.info(f"Columns: {', '.join(df.columns)}")

            # logging null value counts
            null_count = df.isnull().sum()
            logger.info("Number of Nulls per column:")
            for col, count in null_count.items():
                logger.info(f"{col}: {count} null values")

            # data types
            logger.info(f"Data Types: {df.dtypes}")
            
            # data stats
            logger.info(f"Columns summary: {df.describe().to_string()}")
            return df 
        except Exception as e:
            logger.error(f"Error occurred while retrieving data: {str(e)}")
            raise

    @task
    def preprocessing(df, dirs):
        """Preprocesses the input data and saves intermediate files"""
        df = df.copy()
        temp_dir = dirs['temp_dir']
        os.makedirs(temp_dir, exist_ok=True) 
        
        features = ['age', 'contract_length', 'monthly_charges', 'contract_type']
        target = 'churn'
        
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        numerical_features = ['age', 'contract_length', 'monthly_charges']
        categorical_features = ['contract_type']

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

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
        X_test_processed = preprocessor.transform(X_test)

        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train)

        categorical_features_encoded = [f"contract_type_{cat}" for cat in 
                                     preprocessor.named_transformers_['cat']
                                     .named_steps['onehot']
                                     .get_feature_names_out(['contract_type'])]
        feature_names = numerical_features + list(categorical_features_encoded)

        X_train_processed = pd.DataFrame(X_train_balanced, columns=feature_names)
        X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names)

        y_train_df = pd.DataFrame({'target': y_train_balanced.values})
        y_test_df = pd.DataFrame({'target': y_test.values})

        X_train_path = os.path.join(temp_dir, 'X_train.csv')
        X_test_path = os.path.join(temp_dir, 'X_test.csv')
        y_train_path = os.path.join(temp_dir, 'y_train.csv')
        y_test_path = os.path.join(temp_dir, 'y_test.csv')
        preprocessor_path = os.path.join(temp_dir, 'preprocessor.pkl')

        X_train_processed.to_csv(X_train_path, index=False)
        X_test_processed.to_csv(X_test_path, index=False)
        y_train_df.to_csv(y_train_path, index=False)
        y_test_df.to_csv(y_test_path, index=False)

        with open(preprocessor_path, 'wb') as f:
            pickle.dump(preprocessor, f)

        return {
            'X_train_path': X_train_path,
            'X_test_path': X_test_path,
            'y_train_path': y_train_path,
            'y_test_path': y_test_path,
            'preprocessor_path': preprocessor_path
        }

    @task 
    def train_model(paths, dirs):
        """Trains the logistic regression model"""
        X_train = pd.read_csv(paths['X_train_path'])
        y_train = pd.read_csv(paths['y_train_path'])['target']

        logistic_reg = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )

        logistic_reg.fit(X_train, y_train)

        model_path = os.path.join(dirs['temp_dir'], 'model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(logistic_reg, f)

        return model_path

    @task
    def evaluate_model(model_path, data_paths):
        """Evaluates the trained model"""

        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    
        X_test = pd.read_csv(data_paths['X_test_path'])
        y_test = pd.read_csv(data_paths['y_test_path'])['target']

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)

        logger = LoggingMixin().log
        logger.info(f"Model Accuracy: {accuracy}")
        logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")

        return {
            'accuracy': float(accuracy),
            'classification_report': class_report
        }

    @task
    def save_model(model_path, preprocessor_path, dirs):

        """Saves the final model and preprocessor"""

        model_dir = dirs['model_dir']
        final_model_path = os.path.join(model_dir, 'logistic_regression_churn_model.pkl')
        final_preprocessor_path = os.path.join(model_dir, 'churn_preprocessor.pkl')

        with open(model_path, 'rb') as f_in:
            model = pickle.load(f_in)
            with open(final_model_path, 'wb') as f_out:
                pickle.dump(model, f_out)

        with open(preprocessor_path['preprocessor_path'], 'rb') as f_in:
            preprocessor = pickle.load(f_in)
            with open(final_preprocessor_path, 'wb') as f_out:
                pickle.dump(preprocessor, f_out)

        logger = LoggingMixin().log
        logger.info(f"Model saved to {final_model_path}")
        logger.info(f"Preprocessor saved to {final_preprocessor_path}")
        
        return {
            'model_path': final_model_path,
            'preprocessor_path': final_preprocessor_path
        }

    # setting up workflow
    dirs = setup_directories()
    df = get_data()
    processed_data = preprocessing(df, dirs)
    trained_model = train_model(processed_data, dirs)
    evaluation = evaluate_model(trained_model, processed_data)
    final_model = save_model(trained_model, processed_data, dirs)

    # Defining task dependencies
    dirs >> df >> processed_data >> trained_model >> evaluation >> final_model

dag = model_workflow()