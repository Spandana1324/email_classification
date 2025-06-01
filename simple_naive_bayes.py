import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle

def preprocess_text(text):
    """Simple text preprocessing"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs and email addresses
    text = ' '.join([word for word in text.split() if '@' not in word and 'http' not in word])
    
    return text

def train_model():
    """Train a simple Naive Bayes model"""
    print("Loading dataset...")
    df = pd.read_csv("data/email_dataset.csv")
    
    # Basic preprocessing
    print("Preprocessing data...")
    df['processed_email'] = df['email'].apply(preprocess_text)
    
    # Split data
    X = df['processed_email'].values
    y = df['type'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train vectorizer
    print("Training vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.95
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # Train Naive Bayes
    print("Training Naive Bayes model...")
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model and vectorizer
    print("\nSaving model...")
    with open('simple_nb_model.pkl', 'wb') as f:
        pickle.dump((vectorizer, model), f)
    print("Model saved as 'simple_nb_model.pkl'")
    
    return vectorizer, model

def predict_email(email_text, vectorizer, model):
    """Predict the class of a single email"""
    # Preprocess the email
    processed_text = preprocess_text(email_text)
    
    # Transform the text
    text_tfidf = vectorizer.transform([processed_text])
    
    # Make prediction
    prediction = model.predict(text_tfidf)[0]
    probability = model.predict_proba(text_tfidf).max()
    
    return prediction, probability

if __name__ == "__main__":
    # Train the model
    vectorizer, model = train_model()
    
    # Example prediction
    test_email = "This is a test email to check the model's prediction"
    prediction, confidence = predict_email(test_email, vectorizer, model)
    print(f"\nTest prediction: {prediction} (confidence: {confidence:.2f})") 