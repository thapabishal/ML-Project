import sys
import os
import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        """Initializes the prediction pipeline by loading the saved model and preprocessor."""
        try:
            #  Best Practice: Use os.path.join for cross-OS compatibility 
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            
            # Load the artifacts into memory only once
            self.model = load_object(file_path=model_path)
            self.preprocessor = load_object(file_path=preprocessor_path)

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features):
        """Scales the input features and returns the model prediction."""
        try:
            # 1. Transform the data using the fitted preprocessor
            data_scaled = self.preprocessor.transform(features)
            
            # 2. Make the prediction
            pred = self.model.predict(data_scaled)
            
            return pred[0] # Return the first (and only) prediction

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 age: float,
                 study_hours_per_day: float,
                 social_media_hours: float,
                 netflix_hours: float,
                 attendance_percentage: float,
                 sleep_hours: float,
                 exercise_frequency: float,
                 mental_health_rating: float,
                 
                 gender: str,
                 part_time_job: str,
                 diet_quality: str,
                 parental_education_level: str,
                 internet_quality: str,
                 extracurricular_participation: str):
        """Initializes raw data input (14 features)."""
        # Numerical Inputs (8)
        self.age = age
        self.study_hours_per_day = study_hours_per_day
        self.social_media_hours = social_media_hours
        self.netflix_hours = netflix_hours
        self.attendance_percentage = attendance_percentage
        self.sleep_hours = sleep_hours
        self.exercise_frequency = exercise_frequency
        self.mental_health_rating = mental_health_rating
        
        # Categorical Inputs (6)
        self.gender = gender
        self.part_time_job = part_time_job
        self.diet_quality = diet_quality
        self.parental_education_level = parental_education_level
        self.internet_quality = internet_quality
        self.extracurricular_participation = extracurricular_participation


    def get_data_as_data_frame(self):
        """
        Converts the raw data into a DataFrame, performs feature engineering 
        to add the required two features, and returns the final DataFrame (16 features).
        """
        try:
            #  Create dictionary from raw inputs (14 features)
            custom_data_input_dict = {
                "age": [self.age],
                "gender": [self.gender],
                "study_hours_per_day": [self.study_hours_per_day],
                "social_media_hours": [self.social_media_hours],
                "netflix_hours": [self.netflix_hours],
                "part_time_job": [self.part_time_job],
                "attendance_percentage": [self.attendance_percentage],
                "sleep_hours": [self.sleep_hours],
                "diet_quality": [self.diet_quality],
                "exercise_frequency": [self.exercise_frequency],
                "parental_education_level": [self.parental_education_level],
                "internet_quality": [self.internet_quality],
                "mental_health_rating": [self.mental_health_rating],
                "extracurricular_participation": [self.extracurricular_participation],
            }
            
            df = pd.DataFrame(custom_data_input_dict)

            #  AUTOMATED FEATURE ENGINEERING (Adding 2 features) 
            df['total_screen_time'] = df['social_media_hours'] + df['netflix_hours']
            
            # Using epsilon to prevent division by zero
            epsilon = 1e-6
            df['study_sleep_ratio'] = df['study_hours_per_day'] / (df['sleep_hours'] + epsilon)

            # Handle 'None' in Parental Education (to match DataTransformation)
            df['parental_education_level'] = df['parental_education_level'].replace('None', "Unknown")
            
            return df

        except Exception as e:
            raise CustomException(e, sys)

