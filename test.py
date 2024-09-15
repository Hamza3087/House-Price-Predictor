"""
Module for evaluating the linear regression model and running tests.
"""
import os
import pickle
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
        raise FileNotFoundError(
            "Model not found! Please train the model by running 'main.py' first."
        )
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    house_sizes, house_prices = load_data_from_csv(TEST_CSV_FILE_PATH)
    predictions = model.predict(house_sizes)
    mse = mean_squared_error(house_prices, predictions)
    r2 = r2_score(house_prices, predictions)
    print("Model evaluation completed.")
    print(f"Mean Squared Error: {mse}")
    print(f"R2 Score: {r2}")
    return mse, r2

# Pytest functions
def test_load_data_from_csv():
    """Test if data is loaded correctly from CSV."""
    house_sizes, house_prices = load_data_from_csv(TEST_CSV_FILE_PATH)
    assert len(house_sizes) > 0
    assert len(house_prices) > 0
    assert len(house_sizes) == len(house_prices)

def test_model_file_exists():
    """Test if the model file exists."""
    assert os.path.exists(MODEL_PATH), "Model file does not exist"

def test_evaluate_model():
    """Test if the model evaluation runs without errors and returns expected types."""
    mse, r2 = evaluate_model()
    assert isinstance(mse, float)
    assert isinstance(r2, float)
    assert 0 <= r2 <= 1, "R2 score should be between 0 and 1"

if __name__ == "__main__":
    evaluate_model()
