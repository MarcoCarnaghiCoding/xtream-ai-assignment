from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
import logging
from src.pipelines.predict_pipeline import PredictPipeline, CustomData
from src.pipelines.train_pipeline import TrainPipeline, CustomTrainData

app = Flask('Xtream Diamond Price Prediction')



# Set up the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a log folder if it doesn't exist
log_dir = 'log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


# Set up the file handler
handler = logging.FileHandler(filename=os.path.join(log_dir, 'XtreamAPI.log'), mode='a')
handler.setLevel(logging.INFO)
# Set up the formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
# Add the handler to the logger
logger.addHandler(handler)


logger.info("API started")
# -----------------------------------------------------------------------------------
#                            Health check
# -----------------------------------------------------------------------------------
@app.get("/")
def home():
    return{"Health_check": "OK"} 

@app.route('/predict', methods=['POST'])
def predict():
    logger.info("Processing request")

    try:
        # Get input data from request
        input_data = request.get_json()

        # Log the input features
        logger.info("Input features: %s", input_data)

        # Check if the input is a single set of features or a list of sets
        if isinstance(input_data, list):

            logger.info("List request received. Processing input...")

            # Handle multiple sets of features
            features_list = []
            for data in input_data:
                features = CustomData(
                    carat=data['carat'],
                    cut=data['cut'],
                    color=data['color'],
                    clarity=data['clarity'],
                    depth=data['depth'],
                    table=data['table'],
                    x=data['x'],
                    y=data['y'],
                    z=data['z']
                )
                features_list.append(features.get_data_as_data_frame())
            features_df = pd.concat(features_list, ignore_index=True)
        else:

            logger.info("Single request received. Processing input...")

            # Handle a single set of features
            features = CustomData(
                carat=input_data['carat'],
                cut=input_data['cut'],
                color=input_data['color'],
                clarity=input_data['clarity'],
                depth=input_data['depth'],
                table=input_data['table'],
                x=input_data['x'],
                y=input_data['y'],
                z=input_data['z']
            )
            features_df = features.get_data_as_data_frame()

        # Make predictions
        prediction_pipeline = PredictPipeline()
        predictions = prediction_pipeline.predict(features = features_df)

        # Return the predictions
        result = {'predicted_prices': predictions.tolist()} if len(predictions) > 1 else {'predicted_price': float(predictions)}

        # Log the predictions
        logger.info("Predictions: %s", result)

        logger.info("Prediction complete")
        return jsonify(result)

    except Exception as e:
        logger.exception("An error occurred: %s", str(e))
        return jsonify({'error': str(e)})


@app.route("/train", methods=['POST'])
def train():
    logger.info("Processing training request")

    try:
        # Get input data from request
        input_data = request.get_json()

        # Log the input features
        logger.info("Input features: %s", input_data)

        # Check if the input is a single set of features or a list of sets
        if isinstance(input_data, list):

            logger.info("List request received. Processing input...")

            # Handle multiple sets of features
            features_list = []
            for data in input_data:
                features = CustomTrainData(
                    carat=data['carat'],
                    cut=data['cut'],
                    color=data['color'],
                    clarity=data['clarity'],
                    depth=data['depth'],
                    table=data['table'],
                    x=data['x'],
                    y=data['y'],
                    z=data['z'],
                    price=data['price']
                )
                features_list.append(features.get_data_as_data_frame())
            features_df = pd.concat(features_list, ignore_index=True)
        else:

            logger.info("Single request received. Processing input...")

            # Handle a single set of features
            features = CustomTrainData(
                carat=input_data['carat'],
                cut=input_data['cut'],
                color=input_data['color'],
                clarity=input_data['clarity'],
                depth=input_data['depth'],
                table=input_data['table'],
                x=input_data['x'],
                y=input_data['y'],
                z=input_data['z'],
                price=input_data['price']
            )
            features_df = features.get_data_as_data_frame()

        # Check if the 'price' feature is present
        if 'price' not in features_df.columns:
            logger.error("The 'price' feature is missing from the input data.")
            return jsonify({'error': "The 'price' feature is missing from the input data."})

        # Train the model
        training_pipeline = TrainPipeline()
        training_pipeline.train(features_df)

        # Return a success message
        result = {'message': f"The model has been partially trained with {features_df.shape[0]} samples successfully."}
        logger.info("Training complete")
        return jsonify(result)

    except Exception as e:
        logger.exception("An error occurred: %s", str(e))
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)