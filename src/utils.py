import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


import pickle
import numpy as np


# Load environment variables
load_dotenv()

dbname= os.getenv("dbname")
user = os.getenv("user")
password = os.getenv("password")
host = os.getenv("host")
port = os.getenv("port")

def read_PostgreSQL_data():
    logging.info("Reading posgres database started")
    try:
        with psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port,
        ) as mydb:
            logging.info(f"Connection established with Postgres: {mydb}")
            df = pd.read_sql_query("SELECT * FROM student_habits_performance", mydb)
            print(df.head())
            return df


    except Exception as ex:
        raise CustomException(ex)
    
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

            print(f"✅ Training Complete for Model: {list(models.keys())[i]}.")

        return report
    

    except Exception as e:
        raise CustomException(e, sys)