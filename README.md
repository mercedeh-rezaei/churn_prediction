This project impelments an automated machine learning pipeline for predicting customer churn using Apache Airflow. The pipeline includes data retrieval, data processing, model training, evaluation and saving through Airflow DAGs. 

Prereq:
Docker and Docker Compose
Python
PostgresSQL

Steps to run the code:
1. git clone <url>
cd Churn_Prediction

2. Create a .env file with the database credentials:
POSTGRES_USER=chosen_username
POSTGRES_PASSWORD=chosen_password
POSTGRES_DB=churn_prediction
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
AIRFLOW_UID=50000

3. docker-compose up -d

4. Access the airflow UI through http://localhost:8080
setup the PostgresSQL connection in Airflow in Admin/connections

5. Trigger DAG