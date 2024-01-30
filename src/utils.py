import os
import sys
import numpy as np 
import pandas as pd
import pickle
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def feature_engineering(df):
    """
	Performs feature engineering on the input dataframe by calculating beta and alpha values,
	volume, and density of the diamond, and then dropping the auxiliary columns. 
	Returns the modified dataframe.
	"""
    # Calculate the beta and alpha values
    df['beta'] = df['depth'] / 100
    df['alpha'] = (1 - df['beta']) * (1 + (df['table'] / 100)**2)

    # Calculate the volume of the diamond
    df['volume'] = 0.5 * df['z'] * df['x'] * df['y'] * (df['alpha'] + df['beta'])

    # Calculate the density of the diamond
    df['density'] = df['carat'] / df['volume']

    # Drop the auxiliary columns
    df.drop(['beta', 'alpha'], axis=1, inplace=True)

    return df


def removing_outliers(df):
    """
    Remove outliers from the input dataframe based on specific conditions.
    
    Parameters:
    df (DataFrame): The input dataframe containing the data to be processed.
    
    Returns:
    DataFrame: The dataframe with outliers removed.
    """
    # Define the conditions for removing outliers (updated without {price} column)
    conditions = [
        (df['z'] < 2),
        (df['y'] < 2),
        (df['x'] < 2),
        (df['table'] > 75),
        (df['depth'] < 50),
        (df['density'] < 0.008)

    ]

    # Create a mask for the rows to be removed
    mask = np.any(conditions, axis=0)

    # Drop the rows that meet the conditions
    df = df[~mask]
    return df


def drop_redundant_features(df,redundant_features = ['x', 'y', 'z', 'density']):
    """
    Drop redundant features from the input dataframe.

    Parameters:
    df (DataFrame): Input dataframe.
    redundant_features (list): List of feature names to be dropped. Default is ['x', 'y', 'z', 'density'].

    Returns:
    DataFrame: Dataframe with redundant features dropped.
    """
    df = df.drop(redundant_features, axis=1)
    
    return df  


def preprocess_data_to_predict( df,preprocessor,
                                numeric_features = ['volume', 'carat', 'depth', 'table'],
                                categorical_features = ['color', 'cut', 'clarity']
                                ):

    """
	Preprocess the input data for prediction using a given preprocessor.

	Args:
	    df (DataFrame): The input dataframe to be preprocessed.
	    preprocessor: The preprocessor to be used for data transformation.
	    numeric_features (list): List of numeric features to be included in preprocessing.
	    categorical_features (list): List of categorical features to be included in preprocessing.
	    target (str): The target variable for prediction.

	Returns:
	    X_preprocessed: The preprocessed input data for prediction.
	"""
    # Adding Features
    df = feature_engineering(df)

    # Removing Outliers
    df = removing_outliers(df)

    # Drop redundant features
    df = drop_redundant_features(df) 

    # Preprocess the data
    X_preprocessed = preprocessor.transform(df)

    return X_preprocessed