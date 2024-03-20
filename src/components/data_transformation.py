import sys
import os
from dataclasses import dataclass
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from exception import CustomException
from logger import logging
import os
from components.model_trainer import ModelTrainerConfig
from components.model_trainer import ModelTrainer

from utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            train_arr = train_df
            test_arr = test_df

            

            return (
                train_arr,
                test_arr
                
            )
        except Exception as e:
            raise CustomException(e,sys)