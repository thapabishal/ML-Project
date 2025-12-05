from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import sys

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.exception import CustomException

application = Flask(__name__)
app = application


# Route for a home page


@app.route("/")
def index():
    return render_template('home.html', results=None)


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        # Render the form on GET, passing 'results=None'
        return render_template("home.html", results = None, form_data=request.form)

    else:
        try:
            data = CustomData(
                # Numerical Features (Must be converted to float)
                age=float(request.form.get("age")),
                study_hours_per_day=float(request.form.get("study_hours_per_day")),
                social_media_hours=float(request.form.get("social_media_hours")),
                netflix_hours=float(request.form.get("netflix_hours")),
                attendance_percentage=float(request.form.get("attendance_percentage")),
                sleep_hours=float(request.form.get("sleep_hours")),
                exercise_frequency=float(request.form.get("exercise_frequency")),
                mental_health_rating=float(request.form.get("mental_health_rating")),
                # Categorical Features (Strings)
                gender=request.form.get("gender"),
                part_time_job=request.form.get("part_time_job"),
                diet_quality=request.form.get("diet_quality"),
                parental_education_level=request.form.get("parental_education_level"),
                internet_quality=request.form.get("internet_quality"),
                extracurricular_participation=request.form.get(
                    "extracurricular_participation"
                ),
            )
        except Exception as e:
            raise CustomException(e, sys)

        # Prediction pipeline Execution

        # Get the DataFrame (includes the 2 engineered features)
        pred_df = data.get_data_as_data_frame()
        print("Input DataFrame for Prediction:")
        print(pred_df)

        # Initilize prediction pipeline
        predict_pipeline = PredictPipeline()

        # Make Prediction
        results = predict_pipeline.predict(pred_df)

        # Render Result
        return render_template("home.html", results=results)


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,  debug=True)
