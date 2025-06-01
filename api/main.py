from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.masking import PIIMasker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Initialize the masker
masker = PIIMasker()

# Load and prepare the dataset
def load_and_prepare_data():
    try:
        # Load the dataset
        df = pd.read_csv("data/email_dataset.csv")
        
        # Use the correct column names
        X = df['email'].values  # email content
        y = df['type'].values   # email type/category
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None, None, None, None

# Train and evaluate the model
def train_model():
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    if X_train is None:
        print("Could not load dataset. Using empty model.")
        return TfidfVectorizer(), MultinomialNB()
    
    # Create and train the vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # Train the classifier
    classifier = MultinomialNB()
    classifier.fit(X_train_tfidf, y_train)
    
    # Evaluate the model
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = classifier.predict(X_test_tfidf)
    
    # Print evaluation metrics
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return vectorizer, classifier

# Initialize or load the model
if not os.path.exists("models/email_classifier.pkl"):
    os.makedirs("models", exist_ok=True)
    print("Training new model...")
    vectorizer, classifier = train_model()
    # Save the model
    with open("models/email_classifier.pkl", "wb") as f:
        pickle.dump((vectorizer, classifier), f)
else:
    print("Loading existing model...")
    with open("models/email_classifier.pkl", "rb") as f:
        vectorizer, classifier = pickle.load(f)

# FastAPI app
app = FastAPI(title="Email Classification API")

# Input schema
class EmailInput(BaseModel):
    input_email_body: str

# Output schema
class MaskedEntity(BaseModel):
    position: List[int]
    classification: str
    entity: str

class EmailOutput(BaseModel):
    input_email_body: str
    list_of_masked_entities: List[MaskedEntity]
    masked_email: str
    category_of_the_email: str

# API endpoint
@app.post("/classify", response_model=EmailOutput)
async def classify_email(email_input: EmailInput):
    try:
        email_body = email_input.input_email_body

        # Call the masking function
        masked_email, masked_entities = masker.mask_text(email_body)

        # Classify using the trained model
        X = vectorizer.transform([masked_email])
        category = classifier.predict(X)[0]
        
        # Get prediction probability
        proba = classifier.predict_proba(X)[0]
        confidence = np.max(proba)

        # Construct response
        return {
            "input_email_body": email_body,
            "list_of_masked_entities": masked_entities,
            "masked_email": masked_email,
            "category_of_the_email": f"{category} (confidence: {confidence:.2f})"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional: if running directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860, reload=True)