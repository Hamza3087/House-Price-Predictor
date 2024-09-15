"""
Module for serving the linear regression model using Flask.
"""
import os  # Standard library imports
import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np  # Third-party imports

app = Flask(__name__)

# Define the path for the saved model
MODEL_PATH = 'model.pkl'

# Load the trained model
try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as file:
            MODEL = pickle.load(file)
    else:
        raise FileNotFoundError(
            "Model not found! Please train the model by running 'main.py' first."
        )
except (FileNotFoundError, IOError, pickle.UnpicklingError) as e:
    MODEL = None
    print(f"Error loading model: {e}")

@app.route('/')
def home():
    """
    Render the home page.
    """
    return render_template('index.html')

@app.route('/predict_price/', methods=['POST'])
def predict_price():
    """
    Predict house price based on features provided in the POST request.

    Returns:
        json: Predicted price or error message.
    """
    if not MODEL:
        return jsonify({"error": "Model not loaded. Please train the model first."}), 500

    try:
        # Get the house details from the POST request data
        data = request.get_json()

        # Extract and validate features from the request
        size = float(data.get('size', 0))
        bedrooms = int(data.get('bedrooms', 0))
        bathrooms = int(data.get('bathrooms', 0))

        # Ensure valid feature values
        if size <= 0 or bedrooms < 0 or bathrooms < 0:
            raise ValueError("Invalid input values for size, bedrooms, or bathrooms.")

        features = np.array([[size, bedrooms, bathrooms]])

        # Make prediction
        predicted_price = MODEL.predict(features)[0]

        return jsonify({"predicted_price": predicted_price})

    except (ValueError, KeyError) as e:
        return jsonify({"error": f"Value error: {str(e)}"}), 400

    except (FileNotFoundError, IOError, pickle.UnpicklingError) as e:
        return jsonify({"error": f"Model error: {str(e)}"}), 500

    except (TypeError, AttributeError) as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
