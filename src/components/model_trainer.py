import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from urllib.parse import urlparse
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object
# import mlflow

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np 


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def eval_metrics(self, actual, pred):
        """Calculate RMSE, MAE, and R2."""
        mse = mean_squared_error(actual, pred)  # always returns MSE
        rmse = np.sqrt(mse)  # take square root manually
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbours Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # uncomment hyperparameters for better results but it will take more time so commented for now
            params = {
                "Random Forest": {
                    "n_estimators": [50, 100, 200],
                    # "max_depth": [None, 5, 10, 20],
                    # "min_samples_split": [2, 5, 10],
                    # "min_samples_leaf": [1, 2, 4],
                    # "bootstrap": [True, False],
                },
                "Decision Tree": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ],
                    # "splitter": ["best", "random"],
                    # "max_depth": [None, 5, 10, 20],
                    # "min_samples_split": [2, 5, 10],
                    # "min_samples_leaf": [1, 2, 4],
                },
                "Gradient Boosting": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.1, 0.05, 0.01],
                    "subsample": [0.6, 0.8, 1.0],
                    # "max_depth": [3, 5, 7],
                    # "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                },
                "Linear Regression": {
                    # No major hyperparameters in sklearn’s LinearRegression
                },
                "K-Neighbours Regressor": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                    # "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                },
                "XGB Regressor": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.1, 0.05, 0.01],
                    # "max_depth": [3, 5, 7],
                    # "subsample": [0.6, 0.8, 1.0],
                    # "colsample_bytree": [0.6, 0.8, 1.0],
                },
                "CatBoosting Regressor": {
                    "iterations": [100, 200, 500],
                    "depth": [4, 6, 8, 10],
                    "learning_rate": [0.1, 0.05, 0.01],
                },
                "AdaBoost Regressor": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.1, 0.05, 0.01, 1.0],
                    # "loss": ["linear", "square", "exponential"],
                },
            }

            model_report: dict = evaluate_models(
                X_train, y_train, X_test, y_test, models, params
            )

            ## To get best model score from dict
            best_model_score = max(model_report.values())

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            print(
                f"Best model is : {best_model_name} with r2 score: {best_model_score}"
            )

            # model_names = list(params.keys())

            # actual_model = ""

            # for model in model_names:
            #     if best_model_name == model:
            #         actual_model = actual_model + model

            # best_params = params[actual_model]

            # mlflow.set_tracking_uri("https://dagshub.com/thapabishal/Data-Science-Project.mlflow")

            # # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            # mlflow.autolog()


            # Train the best model
            best_model.fit(X_train, y_train)

            # Evaluate
            predicted = best_model.predict(X_test)
            rmse, mae, r2 = self.eval_metrics(y_test, predicted)

            # Log custom metrics (optional, because autolog won’t do RMSE/MAE)
            # mlflow.log_metric("rmse", rmse)
            # mlflow.log_metric("mae", mae)
            # mlflow.log_metric("r2", r2)
            # mlflow tuning done here
            # with mlflow.start_run():
            #     predicted_qualities = best_model.predict(X_test)
            #     (rmse, mae, r2) = self.eval_metrics(y_test, predicted_qualities)

            #     mlflow.log_params(best_params)

            #     mlflow.log_metric("rmse", rmse)
            #     mlflow.log_metric("r2", r2)
            #     mlflow.log_metric("mae", mae)

            # # model registry does not work with file store, so saving the model without registering
            # if tracking_url_type_store != "file":
            #     # Register the model
            #     # There are other ways to use the Model Registry, which depends on the
            #     # please refer to the doc for more information:
            #     # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            #     mlflow.sklearn.log_model(
            #         best_model, "model", registered_model_name=actual_model
            #     )
            # else:
            #     mlflow.sklearn.log_model(best_model, "model")

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(
                f"Best model found: {best_model_name} with r2 score: {best_model_score}"
            )

            # Saving the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            # Use best_model for predictions
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
