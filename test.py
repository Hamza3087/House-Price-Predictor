"""
Module for evaluating a trained linear regression model on house price data.
"""
import os
import pickle  # Standard library imports
import pandas as pd  # Third-party imports
from sklearn.metrics import mean_squared_error, r2_score

MODEL_PATH = 'model.pkl'
TEST_CSV_FILE_PATH = 'house_test_data.csv'


def load_data_from_csv(file_path):
    """
    Load dataset from a CSV file.

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
    Evaluate the trained linear regression model using test data.

    Loads the test data from a CSV file, uses the trained model to make predictions,
    and prints evaluation metrics such as Mean Squared Error (MSE) and R-squared (R2).
    
    Raises:
        FileNotFoundError: If the model file is not found.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found! Please train the model by running 'main.py' first.")

    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)

    house_sizes, house_prices = load_data_from_csv(TEST_CSV_FILE_PATH)

    predicted_prices = model.predict(house_sizes)

    mse = mean_squared_error(house_prices, predicted_prices)
    r2 = r2_score(house_prices, predicted_prices)

    print("Model Evaluation Results:")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R2): {r2}")


if __name__ == "__main__":
    print("Starting model evaluation...")
    evaluate_model()
    print("Model evaluation finished.")
