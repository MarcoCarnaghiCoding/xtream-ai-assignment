import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import os
import sys
from src.exception import CustomException
from src.utils import load_object
#from src.pipelines.predict_pipeline import  PredictPipeline
from src.utils import preprocess_data_to_predict


class PredictPipeline:

    def __init__(self):
        pass

    def predict(self,features):
        try: 
            # Define the model and preprocessor paths
            xgb_model_path = 'artifacts/XGRegressorModel_v2.pkl'
            rf_model_path = 'artifacts/RandomForestRegressorModel.pkl'
            preprocessor_path = 'artifacts/preprocessor_predict.pkl'

            XGB_model = load_object(xgb_model_path)
            RF_model = load_object(rf_model_path)
            preprocessor = load_object(preprocessor_path)

            # Preprocess the data
            X_preprocessed = preprocess_data_to_predict(    df  =   features,
                                                            preprocessor =  preprocessor,
                                                            numeric_features = ['volume', 'carat', 'depth', 'table'],
                                                            categorical_features = ['color', 'cut', 'clarity']
                                                        )
            DX = xgb.DMatrix(X_preprocessed)
        
            # Make predictions
            xgb_predictions = XGB_model.predict(DX)
            rf_predictions = RF_model.predict(X_preprocessed)
        
            # Combine predictions (you can choose a different strategy)
            ensemble_prediction = (xgb_predictions + rf_predictions) / 2.0
            
            return ensemble_prediction
    
        except Exception as e:
            raise CustomException(e,sys)
        


class CustomData:
    def __init__(self, carat, cut, color, clarity, depth, table, x, y, z):
        self.carat = float(carat)
        self.cut = cut
        self.color = color
        self.clarity = clarity
        self.depth = float(depth)
        self.table = float(table)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "carat": [self.carat],
                "cut": [self.cut],
                "color": [self.color],
                "clarity": [self.clarity],
                "depth": [self.depth],
                "table": [self.table],
                "x": [self.x],
                "y": [self.y],
                "z": [self.z]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)