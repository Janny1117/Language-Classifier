from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the saved TF-IDF vectorizer and model
tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
model = joblib.load('models/logistic_regression_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict the language of the input text.
    """
    data = request.json
    text = data['query']  # Use 'query' as the input key
    transformed_text = tfidf_vectorizer.transform([text])
    prediction = model.predict(transformed_text)[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)