import logging
from flask import Flask, request, jsonify, Response
import joblib
import os
from werkzeug.utils import secure_filename
import pdfplumber
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
import time

# Configure logging
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Increase the maximum allowed payload size to 16 MB (or adjust as needed)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# Define metrics
REQUEST_COUNT = Counter('flask_app_requests_total', 'Total number of requests')
REQUEST_LATENCY = Gauge('flask_app_request_latency_seconds', 'Request latency in seconds')

# Initialize metrics with a default value
REQUEST_COUNT.inc(0)
REQUEST_LATENCY.set(0)

# Load the saved TF-IDF vectorizer and model
try:
    tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    model = joblib.load('models/logistic_regression_model.pkl')
    logger.info("Models loaded successfully!")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file using pdfplumber."""
    text = ''
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ''  # Handle pages with no text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
    return text

def clean_word(word):
    """Remove punctuation from a word."""
    return re.sub(r'[^\w\s]', '', word)

def analyze_mixed_language(text):
    """
    Analyze mixed language text and calculate the percentage of Malay and English.
    """
    # Split the text into words
    words = text.split()  # Split by whitespace
    words = [word.strip() for word in words if word.strip()]  # Remove empty words

    # Classify each word
    malay_count = 0
    english_count = 0

    logger.info("Word Classification Results:")
    for word in words:
        cleaned_word = clean_word(word)
        transformed_word = tfidf_vectorizer.transform([cleaned_word])
        prediction = model.predict(transformed_word)[0]
        logger.info(f"Word: {word} -> Prediction: {prediction}")

        if prediction == 'MS':
            malay_count += 1
        else:
            english_count += 1

    # Calculate percentages
    total_words = len(words)
    if total_words == 0:
        return {'malay_percentage': 0, 'english_percentage': 0, 'dominant_language': 'Unknown'}

    malay_percentage = (malay_count / total_words) * 100
    english_percentage = (english_count / total_words) * 100

    # Determine the dominant language
    if malay_percentage > english_percentage:
        dominant_language = 'MS'
    else:
        dominant_language = 'EN'

    return {
        'malay_percentage': malay_percentage,
        'english_percentage': english_percentage,
        'dominant_language': dominant_language
    }

@app.route('/')
def home():
    return "Language Detection App is running!"

@app.route('/metrics')
def metrics():
    """Expose Prometheus metrics."""
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.before_request
def before_request():
    """Record request start time."""
    request.start_time = time.time()

@app.after_request
def after_request(response):
    """Record request latency and increment request count."""
    latency = time.time() - request.start_time
    REQUEST_LATENCY.set(latency)
    REQUEST_COUNT.inc()
    logger.info(f"Request count: {REQUEST_COUNT._value.get()}, Latency: {latency} seconds")
    return response

@app.route('/predict', methods=['POST'])
def predict():
    """Handle file upload and language prediction."""
    try:
        logger.info("File upload request received")
        if 'file' not in request.files:
            logger.error("No file uploaded")
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']

        if file.filename == '':
            logger.error("No file selected")
            return jsonify({'error': 'No file selected'}), 400

        if file and allowed_file(file.filename):
            # Save the uploaded file temporarily
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            logger.info(f"File saved to: {file_path}")

            # Extract text from the PDF
            text = extract_text_from_pdf(file_path)
            logger.info(f"Extracted text: {text[:100]}...")  # Log first 100 characters

            # Analyze mixed language content
            analysis_result = analyze_mixed_language(text)
            logger.info(f"Analysis result: {analysis_result}")

            # Move the file to the appropriate folder based on the dominant language
            language_folder = os.path.join(app.config['UPLOAD_FOLDER'], analysis_result['dominant_language'].lower())
            os.makedirs(language_folder, exist_ok=True)
            new_file_path = os.path.join(language_folder, filename)
            os.rename(file_path, new_file_path)
            logger.info(f"File moved to: {new_file_path}")

            # Return the analysis result
            return jsonify({
                'malay_percentage': analysis_result['malay_percentage'],
                'english_percentage': analysis_result['english_percentage'],
                'dominant_language': analysis_result['dominant_language'],
                'file_path': new_file_path
            })

        logger.error("Invalid file type")
        return jsonify({'error': 'Invalid file type'}), 400
    except Exception as e:
        logger.error(f"Error in /predict endpoint: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)