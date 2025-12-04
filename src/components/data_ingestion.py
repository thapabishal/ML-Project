import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from src.utils import read_PostgreSQL_data 


from sklearn.model_selection import train_test_split
from dataclasses import dataclass


from src.components.data_transformation import DataTransformation , DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'data.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Data Ingestion function called")
            ## data ingestion portion code
            df = read_PostgreSQL_data()

            ## datatransformation portion code
            # df= pd.read_csv('notebook/student_habits_performance.csv')

            # Drop the index (student_id) column
            df = df.drop(columns=['student_id'], axis=1) #
            
            logging.info("Reading completed from PostgreSQL database")

            # Ensure artifacts folder exists
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Split into train/test
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save train/test sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data Ingestion is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))