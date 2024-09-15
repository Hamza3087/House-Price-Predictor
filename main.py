"""
Module for training a linear regression model on house price data.
"""
import pickle  # Standard library imports
import pandas as pd  # Third-party imports
from sklearn.linear_model import LinearRegression

MODEL_PATH = 'model.pkl'
CSV_FILE_PATH = 'house_data.csv'


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


def train_model():
    """
    Train a linear regression model on house data and save the model.

    Loads house size and price data from a CSV file, trains a linear regression
    model, and saves the trained model to a file.
    """
    house_sizes, house_prices = load_data_from_csv(CSV_FILE_PATH)

    model = LinearRegression()
    model.fit(house_sizes, house_prices)

    with open(MODEL_PATH, 'wb') as file:
        pickle.dump(model, file)

    print("Model training complete. Model saved to 'model.pkl'.")


if __name__ == "__main__":
    print("Starting model training...")
    train_model()
    print("Model training finished.")
