import pytest
import pickle
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

MODEL_PATH = 'model.pkl'
TEST_CSV_FILE_PATH = 'house_test_data.csv'

def load_data_from_csv(file_path):
    data = pd.read_csv(file_path)
    house_sizes = data[['Size', 'Bedrooms', 'Bathrooms']].values
    house_prices = data['Price'].values
    return house_sizes, house_prices

def evaluate_model():
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
    return mse, r2

def test_load_data_from_csv():
    house_sizes, house_prices = load_data_from_csv(TEST_CSV_FILE_PATH)
    assert len(house_sizes) > 0
    assert len(house_prices) > 0
    assert len(house_sizes) == len(house_prices)

def test_model_file_exists():
    assert os.path.exists(MODEL_PATH), "Model file does not exist"

def test_evaluate_model():
    mse, r2 = evaluate_model()
    assert isinstance(mse, float)
    assert isinstance(r2, float)
    assert 0 <= r2 <= 1, "R2 score should be between 0 and 1"

if __name__ == "__main__":
    pytest.main()
