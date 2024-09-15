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
        raise FileNotFoundError("Model not found! Please train the model by running 'main
