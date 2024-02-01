import requests

"""
Module to generate test cases for the API endpoint.

This module contains examples of single sets of features to be used for testing the API endpoint.

Examples:
    input_data (dict): A single set of features for testing the API with a single request.
    response (Response): The response object for the request with a single set of features.
"""

input_data = {
  "carat": 1.1,
  "cut": "Ideal",
  "color": "H",
  "clarity": "SI2",
  "depth": 62.0,
  "table": 55.0,
  "x": 6.61,
  "y": 6.65,
  "z": 4.11
}


response = requests.post('http://localhost:5000/predict', json=input_data)
print(f'Predicted price: {response.json()["predicted_price"]}') 
print(f'Real price: 4733') 