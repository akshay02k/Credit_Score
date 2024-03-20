import os
import sys
from dataclasses import dataclass
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


from exception import CustomException
from logger import logging
from utils import save_object
from utils import evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()



    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X = np.concatenate((train_array, test_array))[:,:-1]
            y = np.concatenate((train_array, test_array))[:,-1]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            models = {
                
                "Random Forest Classifier":RandomForestClassifier(),
                "XGB Classifier":XGBClassifier(),
                "Decision Tree Classifier":DecisionTreeClassifier(),
                "Catboost Classifeir": CatBoostClassifier()

            }


            logging.info("model_report dictionary is creating")
            model_report:dict = evaluate_models(X_train = X_train, y_train = y_train, X_test=X_test,y_test=y_test, models = models)
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model Found")

            logging.info(f"Best model is found")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            accuracyscore = accuracy_score(y_test, predicted)
            return accuracy_score



        except Exception as e:
            raise CustomException(e, sys)