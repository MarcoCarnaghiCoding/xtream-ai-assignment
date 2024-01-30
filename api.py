from flask import Flask, request, jsonify
import joblib
from src.pipelines.predict_pipeline import PredictPipeline, CustomData

app = Flask(__name__)


## Route for a home page
# -----------------------------------------------------------------------------------
#                            Health check
# -----------------------------------------------------------------------------------
@app.get("/")
def home():
    return{"Health_check": "OK"} 

@app.route('/predict', methods=['POST'])
def predict():
    
    try:
        # Get input data from request
        input_data = request.get_json()
        
        # Preprocess input data
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
        prediction = prediction_pipeline.predict(features = features_df)
        print(prediction)
        # Return the predictions
        result = {'predicted_price': float(prediction)}

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
