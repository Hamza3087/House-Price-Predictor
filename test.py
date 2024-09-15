import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

MODEL_PATH = 'model.pkl'
TEST_CSV_FILE_PATH = 'house_test_data.csv'

def load_data_from_csv(file_path):
    # Load dataset from CSV
    data = pd.read_csv(file_path)
    house_sizes = data[['Size', 'Bedrooms', 'Bathrooms']].values
    house_prices = data['Price'].values
    return house_sizes, house_prices

def evaluate_model():
    # Check if the model exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found! Please train the model by running 'main.py' first.")

    # Load the trained model
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)

    # Load the test data
    house_sizes, house_prices = load_data_from_csv(TEST_CSV_FILE_PATH)
    
    # Make predictions
    predicted_prices = model.predict(house_sizes)
    
    # Calculate performance metrics
    mse = mean_squared_error(house_prices, predicted_prices)
    r2 = r2_score(house_prices, predicted_prices)
    
    print(f"Model Evaluation Results:")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R2): {r2}")

if __name__ == "__main__":
    print("Starting model evaluation...")
    evaluate_model()
    print("Model evaluation finished.")
