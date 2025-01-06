import pandas as pd
import os
import re

def preprocess_data(input_file):
    """
    Load, validate, and preprocess the dataset.
    """
    # Load data
    try:
        data = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
        print("Current Working Directory:", os.getcwd())
        print("Files in Directory:", os.listdir())
        raise

    # Validate data
    required_columns = ['lan_code', 'query']  # Ensure these columns exist
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"Dataset must contain the following columns: {required_columns}")

    # Check for missing values in 'lan_code' and 'query'
    if data['lan_code'].isnull().any() or data['query'].isnull().any():
        print("Warning: Dataset contains missing values in 'lan_code' or 'query' columns.")
        print("Rows with missing values will be dropped.")

    # Drop rows with missing values in 'lan_code' or 'query'
    processed_data = data.dropna(subset=['lan_code', 'query'])

    # Clean text in the 'query' column
    processed_data['query'] = processed_data['query'].fillna("").apply(clean_text)  # Fill NaN with empty string

    # Save the preprocessed data to a new CSV file
    processed_data.to_csv(r"data/preprocessed_data.csv", index=False)

    return processed_data

def clean_text(text):
    """
    Clean text by lowercasing and removing punctuation.
    """
    if not isinstance(text, str):  # Handle non-string values
        return ""
    text = text.lower()  # Lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

if __name__ == "__main__":
    # Define the absolute file path
    file_path = r"C:\Users\User\Downloads\MachineLearning\project\data\Language_Malay_English_dataset.csv"

    # Run the preprocessing function
    data = preprocess_data(file_path)
    print(data.head())  # Print the first few rows of the processed data