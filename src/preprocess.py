import pandas as pd

def preprocess_data(input_file):
    # Load data from CSV file
    data = pd.read_csv(input_file)
    
    # Example of data processing: removing rows with missing values
    processed_data = data.dropna()
    
    return processed_data

# Run the function when the script is executed directly
if __name__ == "__main__":
    # Ensure the file path is correct (use raw string or double backslashes)
    data = preprocess_data(r"C:\Users\User\Downloads\MachineLearning\project\Language( Malay & English )_dataset.csv")
    print(data.head())  # Print the first few rows of the processed data