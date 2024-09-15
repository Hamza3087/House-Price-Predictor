from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import os

app = Flask(__name__)

# Define the path for the saved model
MODEL_PATH = 'model.pkl'

# Load the trained model
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
else:
    raise FileNotFoundError("Model not found! Please train the model by running 'main.py' first.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_price/', methods=['POST'])
def predict_price():
    try:
        # Get the house details from the POST request data
        data = request.get_json()
        size = float(data.get('size'))
        bedrooms = int(data.get('bedrooms'))
        bathrooms = int(data.get('bathrooms'))
        
        features = np.array([[size, bedrooms, bathrooms]])
        predicted_price = model.predict(features)[0]
        return jsonify({"predicted_price": predicted_price})
    
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
