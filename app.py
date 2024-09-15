import os  # Standard library imports
import pickle
import numpy as np  # Third-party imports
from flask import Flask, request, jsonify, render_template

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
    return render_template('index.html')

@app.route('/predict_price/', methods=['POST'])
def predict_price():
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

    except Exception as e:
        return jsonify({
            "error": f"An error occurred: {str(e)}"
        }), 500

if __name__ == "__main__":
    app.run(debug=True)
