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
        "carat": 1.1,
        "cut": "Ideal",
        "color": "H",
        "clarity": "SI2",
        "depth": 62.0,
        "table": 62.0,
        "x": 6.61,
        "y": 6.65,
        "z": 4.11,
        "price": 4733
    }

# Example with a list of sets of features
multiple_features = [
    {
        "carat": 1.29,
        "cut": "Ideal",
        "color": "H",
        "clarity": "SI1",
        "depth": 62.6,
        "table": 56.0,
        "x": 6.96,
        "y": 6.93,
        "z": 4.35,
        "price": 6424
    },
    {
        "carat": 1.1,
        "cut": "Ideal",
        "color": "H",
        "clarity": "SI2",
        "depth": 62.0,
        "table": 62.0,
        "x": 6.61,
        "y": 6.65,
        "z": 4.11,
        "price": 4733
    }
]



# Set the API endpoint
url = 'http://127.0.0.1:5000/train'

# Make a request with a single set of features
response_single = requests.post(url, json=single_features)
print(response_single.json())

# Make a request with a list of sets of features
response_multiple = requests.post(url, json=multiple_features)
print(response_multiple.json())
