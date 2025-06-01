print("API script started.")
from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import hstack
import sys
import os

# Add the directory containing utils to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.masking import PIIMasker
from utils.demasking import PIIDemasker

# Import necessary functions from svm_model.py (assuming it's in the same directory)
from svm_model import clean_text, extract_features, nlp # We need nlp from spacy too

app = Flask(__name__)

# --- Load Model and Vectorizer ---
model_path = 'models/ensemble_model.pkl' # Assuming you saved the ensemble model
vectorizer_path = 'models/vectorizer.pkl'

model = None
vectorizer = None
masker = None
demasker = None

def load_model_and_vectorizer():
    """Loads the trained model, vectorizer, masker, and demasker."""
    global model, vectorizer, masker, demasker
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded successfully from {model_path}")

        with open(vectorizer_path, 'rb') as f:
            loaded_objects = pickle.load(f)
            vectorizer = loaded_objects[0]
            masker = loaded_objects[1]
            demasker = loaded_objects[2] # Assuming masker and demasker are at indices 1 and 2
        print(f"Vectorizer, Masker, and Demasker loaded successfully from {vectorizer_path}")

    except FileNotFoundError:
        print(f"Error: Model or vectorizer file not found. Please ensure '{model_path}' and '{vectorizer_path}' exist.")
        sys.exit(1) # Exit if files are not found
    except Exception as e:
        print(f"Error loading model or vectorizer: {e}")
        sys.exit(1) # Exit on other loading errors

# Load everything when the app starts
with app.app_context():
    load_model_and_vectorizer()

# --- API Endpoint ---
@app.route('/classify', methods=['POST'])
def classify_email():
    # Expecting input in the format: {"input_email_body": "string containing the email"}
    if not request.json or not 'input_email_body' in request.json:
        return jsonify({"error": "Please provide 'input_email_body' in the request body."}), 400

    email_text = request.json['input_email_body']

    if not model or not vectorizer or not masker or not demasker:
         return jsonify({"error": "Model not loaded. Please check server logs."}), 500

    try:
        # 1. Mask PII - Keep the masked_email and list of masked entities
        masked_email, masked_entities = masker.mask_text(email_text)

        # 2. Clean and prepare text (using the cleaned masked email for additional features)
        cleaned_text_from_masked = clean_text(masked_email)

        # 3. Transform the masked email text using the trained vectorizer
        text_features = vectorizer.transform([masked_email])

        # 4. Extract additional features from the cleaned masked text
        additional_features_dict = extract_features(cleaned_text_from_masked)
        additional_features_df = pd.DataFrame([additional_features_dict])

        # 5. Combine text features and additional features
        features = hstack([text_features, additional_features_df])

        # 6. Get prediction
        # The loaded 'model' is the GridSearchCV best estimator pipeline, which handles scaling and ensemble prediction
        predicted_category = str(model.predict(features)[0]) # Ensure category is string
        # The requirement only asks for the category, not confidence or top 3

        # 7. Prepare the response dictionary according to the required format
        response = {
            "input_email_body": email_text,
            "list_of_masked_entities": masked_entities,
            "masked_email": masked_email,
            "category_of_the_email": predicted_category
        }

        return jsonify(response), 200

    except Exception as e:
        # Log the error for debugging
        app.logger.error(f"Error processing email: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred during classification."}), 500

# --- Run the Flask app ---
if __name__ == '__main__':
    # Use a more production-ready server like Waitress or Gunicorn in production
    # For development, debug=True is fine
    app.run(debug=True, port=5000)