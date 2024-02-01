import requests


"""
Module to generate test cases for the API endpoint.

This module contains examples of single and multiple sets of features to be used for testing the API endpoint.

Examples:
    single_features (dict): A single set of features for testing the API with a single request.
    multiple_features (list): A list of sets of features for testing the API with multiple requests.
    url (str): The API endpoint URL.
    response_single (Response): The response object for the request with a single set of features.
    response_multiple (Response): The response object for the request with multiple sets of features.

"""

# Example with a single set of features
single_features = {
    "carat": 0.5,
    "cut": "Ideal",
    "color": "E",
    "clarity": "SI1",
    "depth": 61.5,
    "table": 55.0,
    "x": 5.0,
    "y": 5.1,
    "z": 3.0
}

# Example with a list of sets of features
multiple_features = [
    {
        "carat": 0.5,
        "cut": "Ideal",
        "color": "E",
        "clarity": "SI1",
        "depth": 61.5,
        "table": 55.0,
        "x": 5.0,
        "y": 5.1,
        "z": 3.0
    },
    {
        "carat": 0.7,
        "cut": "Premium",
        "color": "D",
        "clarity": "VS2",
        "depth": 62.3,
        "table": 58.0,
        "x": 5.7,
        "y": 5.8,
        "z": 3.6
    }
]

# Set the API endpoint
url = 'http://127.0.0.1:5000/predict'

# Make a request with a single set of features
response_single = requests.post(url, json=single_features)
print(response_single.json())

# Make a request with a list of sets of features
response_multiple = requests.post(url, json=multiple_features)
print(response_multiple.json())
