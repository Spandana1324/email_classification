import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

def preprocess_data(df):
    """Preprocess the email data"""
    print("\nPreprocessing Data:")
    print(f"Total samples: {len(df)}")
    
    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Remove any missing values
    df = df.dropna()
    print(f"\nSamples after removing missing values: {len(df)}")
    
    # Check class distribution
    print("\nClass distribution:")
    print(df['type'].value_counts())
    
    return df

def test_single_email(email, vectorizer, model):
    """Test a single email and show detailed prediction results"""
    print("\n" + "="*50)
    print("Testing Email:")
    print("-"*50)
    print(f"Email content: {email}")
    print("-"*50)
    
    # Transform the email
    X = vectorizer.transform([email])
    
    # Get prediction and probabilities
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    
    # Get all class labels
    class_labels = model.classes_
    
    # Print results
    print("\nPrediction Results:")
    print("-"*50)
    print(f"Predicted category: {prediction}")
    print("\nConfidence scores for each category:")
    for label, prob in zip(class_labels, probabilities):
        print(f"{label}: {prob:.2%}")
    print("="*50)

def train_and_evaluate():
    """Train and evaluate Random Forest model"""
    print("\nLoading dataset...")
    df = pd.read_csv("data/email_dataset.csv")
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Split features and target
    X = df['email'].values
    y = df['type'].values
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Create and train the vectorizer
    print("\nTraining TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 3)
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train Random Forest model
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train_tfidf, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nRandom Forest Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Random Forest - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('random_forest_confusion_matrix.png')
    print("\nConfusion matrix saved as 'random_forest_confusion_matrix.png'")
    
    # Save the model and vectorizer
    print("\nSaving model and vectorizer...")
    os.makedirs("models", exist_ok=True)
    with open("models/random_forest_model.pkl", "wb") as f:
        pickle.dump((vectorizer, model), f)
    
    # Test with example emails
    print("\nTesting with example emails:")
    test_emails = [
        "Dear customer, your account has been suspended. Please click here to verify your details.",
        "Meeting scheduled for tomorrow at 2 PM in Conference Room A.",
        "Your order #12345 has been shipped and will arrive in 2-3 business days."
    ]
    
    for email in test_emails:
        test_single_email(email, vectorizer, model)
    
    # Interactive testing
    while True:
        print("\n" + "="*50)
        print("Enter your email to test (or 'quit' to exit):")
        user_email = input("> ")
        
        if user_email.lower() == 'quit':
            break
            
        if user_email.strip():
            test_single_email(user_email, vectorizer, model)
        else:
            print("Please enter a valid email.")

if __name__ == "__main__":
    train_and_evaluate() 