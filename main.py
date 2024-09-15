import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import os

MODEL_PATH = 'model.pkl'
CSV_FILE_PATH = 'house_data.csv'

def load_data_from_csv(file_path):
    # Load dataset from CSV
    data = pd.read_csv(file_path)
    house_sizes = data[['Size', 'Bedrooms', 'Bathrooms']].values
    house_prices = data['Price'].values
    return house_sizes, house_prices

def train_model():
    # Load data from CSV
    house_sizes, house_prices = load_data_from_csv(CSV_FILE_PATH)
    
    # Train the Linear Regression model
    model = LinearRegression()
    model.fit(house_sizes, house_prices)
    
    # Save the model to a file
    with open(MODEL_PATH, 'wb') as file:
        pickle.dump(model, file)
    
    print("Model training complete. Model saved to 'model.pkl'.")

if __name__ == "__main__":
    # Train the model when the script is run
    print("Starting model training...")
    train_model()
    print("Model training finished.")
