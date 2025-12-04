import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation
        """
        try:
            numerical_columns = [
                "age",
                "study_hours_per_day",
                "social_media_hours",
                "netflix_hours",
                "attendance_percentage",
                "sleep_hours",
                "exercise_frequency",
                "mental_health_rating",
                "total_screen_time",
                "study_sleep_ratio",
            ]
            categorical_columns = [
                "gender",
                "part_time_job",
                "diet_quality",
                "parental_education_level",
                "internet_quality",
                "extracurricular_participation",
            ]

            # Numerical pipeline for missing values and scaling
            num_pipeline = Pipeline(
                steps=[
                    # Fills any NaN/missing numerical values with the median of the column
                    ("imputer", SimpleImputer(strategy="median")),
                    # Scales the features (essential for linear models, regularization, etc.)
                    ("scaler", StandardScaler()),
                ]
            )

            # Categorical pipeline for missing values, encoding and scaling
            cat_pipeline = Pipeline(
                steps=[
                    # AS we did in EDA, we only have missing categorical value in parental_education_level and we want to seprate it from other so we kept "Unknown"
                    ("imputer", SimpleImputer(strategy="constant", fill_value ='missing')),
                    # Converts categorical strings into numerical binary features
                    # Added 'handle_unknown="ignore"' for robust test data handling.
                    # Added 'sparse_output=False' for easier NumPy array concatenation.
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                    # Not using StandardScaler from cat_pipeline as it's often unnecessary for one-hot encoded features.
                ]
            )

            logging.info(f"Categorical Columns: {categorical_columns}")
            logging.info(f"Numerical Columns: {numerical_columns}")

            # Combines the pipelines using ColumnTransformer
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ],
                remainder='passthrough' # Keep any columns not listed (eg. student_id)
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # FEATURE ENGINEERING (adding total screen time and study to sleep ratio)
            # Create 'total_screen_time'
            train_df['total_screen_time'] = train_df['social_media_hours'] + train_df['netflix_hours']
            test_df['total_screen_time'] = test_df['social_media_hours'] + test_df['netflix_hours']

            # Create 'study_sleep_ratio'
            epsilon = 1e-6 
            train_df['study_sleep_ratio'] = train_df['study_hours_per_day'] / (train_df['sleep_hours'] + epsilon)
            test_df['study_sleep_ratio'] = test_df['study_hours_per_day'] / (test_df['sleep_hours'] + epsilon)
            
            logging.info("Obtaining preprocessor object and feature engineering completed")

            preprocessor_obj = self.get_data_transformer_object()
            

            target_column_name = "exam_score"
            columns_to_drop = [target_column_name]

            # divide the train dataset to independent (X) and dependent (y) features
            input_feature_train_df = train_df.drop(columns=columns_to_drop, axis=1)
            target_feature_train_df = train_df[target_column_name]

            # # divide the train dataset to independent and dependent features
            input_feature_test_df = test_df.drop(columns=columns_to_drop, axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                "Applying preprocessing object on training and testing datasets."
            )

            # Transforming using preprocessor object
            input_feature_train_arr = preprocessor_obj.fit_transform(
                input_feature_train_df
            )
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            # Combining transformed features with target variable
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            # Saving the preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj,
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
        


