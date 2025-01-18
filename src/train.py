import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import joblib
import os

def train_and_save_models(X_train, y_train):
    """
    Train and save Naive Bayes and Logistic Regression models.
    """
    # Train Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)

    # Train Logistic Regression
    lr_model = LogisticRegression(solver='liblinear', max_iter=1000)  # Use liblinear solver
    lr_model.fit(X_train, y_train)

    # Ensure the 'models' directory exists
    os.makedirs("models", exist_ok=True)

    # Save models
    joblib.dump(nb_model, 'models/naive_bayes_model.pkl')
    joblib.dump(lr_model, 'models/logistic_regression_model.pkl')

    return nb_model, lr_model

if __name__ == "__main__":
    # Load the preprocessed data
    data = pd.read_csv("data/preprocessed_data.csv")

    # Split data into features and labels
    X = data['query']  # Use 'query' as the feature column
    y = data['lan_code']  # Use 'lan_code' as the target column

    # Vectorize text data
    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(X)

    # Ensure the 'models' directory exists
    os.makedirs("models", exist_ok=True)

    # Save the TF-IDF vectorizer
    joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')

    # Train and save models
    train_and_save_models(X_tfidf, y)