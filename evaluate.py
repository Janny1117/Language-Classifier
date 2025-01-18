import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
import os

def evaluate_model_performance(data_path, tfidf_vectorizer, nb_model, lr_model):
    try:
        # Debug: Print script status
        print("Evaluating model performance...")

        # Load preprocessed data
        print(f"Loading data from: {data_path}")
        dataset = pd.read_csv(data_path)
        print("Data loaded successfully!")
        print("Columns in data:", dataset.columns)
        print("First few rows of data:\n", dataset.head())

        # Check if required columns exist
        if 'query' not in dataset.columns or 'lan_code' not in dataset.columns:
            raise KeyError("The dataset must contain 'query' and 'lan_code' columns.")

        texts = dataset['query'].astype(str)
        labels = dataset['lan_code']

        # Transform data using the saved TF-IDF vectorizer
        tfidf_data = tfidf_vectorizer.transform(texts)

        # Evaluate Naive Bayes model
        nb_predictions = nb_model.predict(tfidf_data)
        print("Naive Bayes Performance:\n", classification_report(labels, nb_predictions))

        # Evaluate Logistic Regression model
        lr_predictions = lr_model.predict(tfidf_data)
        print("Logistic Regression Performance:\n", classification_report(labels, lr_predictions))

        # Plot confusion matrix for Logistic Regression
        cm = confusion_matrix(labels, lr_predictions, labels=lr_model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lr_model.classes_)
        
        # Create the evaluations directory if it doesn't exist
        os.makedirs("evaluations", exist_ok=True)
        
        # Save the confusion matrix plot
        print("Saving confusion matrix plot...")
        disp.plot()
        plt.savefig("evaluations/confusion_matrix.png")
        plt.close()  # Close the plot to free resources
        print("Confusion matrix plot saved successfully!")

    except FileNotFoundError:
        print(f"Error: File not found at '{data_path}'. Please check the file path.")
    except KeyError as e:
        print(f"Error: Column '{e}' not found in the dataset. Please check the column names.")
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{data_path}' is empty or does not contain valid CSV data.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Debug: Print script status
    print("Script started!")

    # Load the saved TF-IDF vectorizer and models
    try:
        tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        print("TF-IDF Vectorizer loaded successfully!")
        nb_model = joblib.load('models/naive_bayes_model.pkl')
        print("Naive Bayes model loaded successfully!")
        lr_model = joblib.load('models/logistic_regression_model.pkl')
        print("Logistic Regression model loaded successfully!")
    except Exception as e:
        print(f"Error loading models or vectorizer: {e}")
        exit()

    # Evaluate model performance on the preprocessed data
    data_path = r"data/preprocessed_data.csv"
    evaluate_model_performance(data_path, tfidf_vectorizer, nb_model, lr_model)