"""
Module for evaluating the linear regression model.
"""
import pickle  # Standard library imports
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

MODEL_PATH = 'model.pkl'
TEST_CSV_FILE_PATH = 'house_test_data.csv'

def load_data_from_csv(file_path):
    """
    Load dataset from CSV.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        tuple: A tuple containing the house sizes and prices.
    """
    data = pd.read_csv(file_path)
    house_sizes = data[['Size', 'Bedrooms', 'Bathrooms']].values
    house_prices = data['Price'].values
    return house_sizes, house_prices

def evaluate_model():
    """
    Evaluate the model's performance on test data.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found! Please train the model by running 'main.py' first.")
    
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)

    house_sizes, house_prices = load_data_from_csv(TEST_CSV_FILE_PATH)
    predictions = model.predict(house_sizes)

    mse = mean_squared_error(house_prices, predictions)
    r2 = r2_score(house_prices, predictions)

    # Breaking long line into shorter lines
    print(f"Model evaluation completed. Mean Squared Error: {mse}, "
          f"R2 Score: {r2}")

if __name__ == "__main__":
    evaluate_model()
