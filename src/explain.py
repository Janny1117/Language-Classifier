import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

def explain_model():
    """
    Generate SHAP explainability plots.
    """
    # Load the saved TF-IDF vectorizer and model
    tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    model = joblib.load('models/logistic_regression_model.pkl')

    # Load training data for background
    data = pd.read_csv(r"data/Language_Malay_English_dataset.csv")
    X_train = tfidf_vectorizer.transform(data['query'])

    # Prepare background data for SHAP explainer
    background_data = shap.utils.sample(X_train, 100)  # Use a sample of training data

    # Create the SHAP explainer
    explainer = shap.LinearExplainer(model, background_data, feature_perturbation="interventional")

    # Example text for explanation
    example_text = ["Family"]
    transformed_example = tfidf_vectorizer.transform(example_text)

    # Get SHAP values for the example
    shap_values = explainer.shap_values(transformed_example)

    # Plot the SHAP summary plot
    shap.summary_plot(shap_values, feature_names=tfidf_vectorizer.get_feature_names_out(), plot_type="bar")
    plt.savefig("evaluations/shap_summary.png")

if __name__ == "__main__":
    explain_model()