import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import os
import sys
import logging
from src.exception import CustomException
from src.utils import load_object, save_object
#from src.pipelines.predict_pipeline import  PredictPipeline
from src.utils import preprocess_data_to_train
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# Initialize the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Set up the file handler
log_dir = 'log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
handler = logging.FileHandler(filename=os.path.join(log_dir, 'TrainPipeline.log'), mode='a')
handler.setLevel(logging.INFO)

# Set up the formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)


class TrainPipeline:

    def __init__(self):
        pass

    def train(self,features):
        try:    
            logger.info("Training pipeline started")

            # Define the model and preprocessor paths
            xgb_model_path = 'artifacts/XGRegressorModel_v2.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'

            XGB_model = load_object(xgb_model_path)
            preprocessor = load_object(preprocessor_path)

            # Preprocess the data
            X_new_preprocessed, y_new  = preprocess_data_to_train(      df  =   features,
                                                                        preprocessor =  preprocessor,
                                                                        numeric_features = ['volume', 'carat', 'depth', 'table'],
                                                                        categorical_features = ['color', 'cut', 'clarity'],
                                                                        target = 'price'
                                                                        )
            dnew = xgb.DMatrix(X_new_preprocessed, label=y_new)
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'eta': 0.1,
            }
            params['nrounds'] = 0
            updated_model = xgb.train(params, dnew, xgb_model=XGB_model, num_boost_round=0)
            
            # Evaluate the performance of the updated model
            logger.info("Evaluating model performance")
            xgb_predictions = updated_model.predict(dnew)
            xgb_rmse = mean_squared_error(y_new, xgb_predictions, squared=False)
            xgb_r2 = r2_score(y_new, xgb_predictions)

            logger.info("XGBoost Metrics:")
            logger.info(f"Root Mean Squared Error (RMSE): {xgb_rmse:.2f}")
            logger.info(f"R-squared (R2): {xgb_r2:.2f}")
            
            # Overwrite the old model with the updated model
            logger.info('Overwriting old model with updated model')
            save_object(xgb_model_path, updated_model)

            logger.info('Pipeline execution completed')

            return None
    
        except Exception as e:
            raise CustomException(e,sys)
        


class CustomTrainData:
    def __init__(self, carat, cut, color, clarity, depth, table, x, y, z, price):
        self.carat = float(carat)
        self.cut = cut
        self.color = color
        self.clarity = clarity
        self.depth = float(depth)
        self.table = float(table)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.price = float(price)

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
                "z": [self.z],
                "price": [self.price]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)