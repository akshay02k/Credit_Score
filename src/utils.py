import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import accuracy_score
from logger import logging
import pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from exception import CustomException

def save_object(file_path, obj):
    try:

        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:

        report = {}
        for model_name, model_instance in models.items():
            # Fit the model on the training data
            model_instance.fit(X_train, y_train)
            
            # Make predictions on both training and testing data
            y_train_pred = model_instance.predict(X_train)
            y_test_pred = model_instance.predict(X_test)

            # Calculate accuracy scores
            train_model_score = accuracy_score(y_train, y_train_pred)
            test_model_score = accuracy_score(y_test, y_test_pred)

            # Log the information
            logging.info(f"Model: {model_name}, Train Accuracy: {train_model_score}, Test Accuracy: {test_model_score}")

            # Store the test model score in the report
            report[model_name] = test_model_score

        return report
    except Exception as e:
        raise CustomException(e,sys)