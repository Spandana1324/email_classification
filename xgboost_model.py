import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from tqdm import tqdm

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

def test_single_email(email, vectorizer, model, label_encoder):
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
    
    # Convert numeric prediction back to original label
    original_prediction = label_encoder.inverse_transform([prediction])[0]
    
    # Get all class labels
    class_labels = label_encoder.classes_
    
    # Print results
    print("\nPrediction Results:")
    print("-"*50)
    print(f"Predicted category: {original_prediction}")
    print("\nConfidence scores for each category:")
    for label, prob in zip(class_labels, probabilities):
        print(f"{label}: {prob:.2%}")
    print("="*50)

def train_and_evaluate():
    """Train and evaluate XGBoost model"""
    print("\nLoading dataset...")
    df = pd.read_csv("data/email_dataset.csv")
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Initialize label encoder
    label_encoder = LabelEncoder()
    
    # Split features and target
    X = df['email'].values
    y = df['type'].values
    
    # Encode the target labels
    y_encoded = label_encoder.fit_transform(y)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Create and train the vectorizer with reduced features
    print("\nTraining TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=2000,  # Further reduced from 5000
        min_df=5,          # Increased from 3
        max_df=0.95,
        ngram_range=(1, 2)  # Keep only unigrams and bigrams
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Create XGBoost model with CPU settings
    print("\nTraining XGBoost model...")
    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,        # Reduced from 6
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,         # Use all CPU cores
        tree_method='hist', # Use histogram-based algorithm
        gpu_id=-1          # Disable GPU
    )
    
    # Simplified parameter grid
    param_grid = {
        'n_estimators': [100],
        'max_depth': [4],
        'learning_rate': [0.1],
        'min_child_weight': [1],
        'subsample': [0.8],
        'colsample_bytree': [0.8]
    }
    
    # Create GridSearchCV object with reduced cv folds
    grid_search = GridSearchCV(
        xgb_model,
        param_grid,
        cv=2,  # Reduced from 3
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Train the model with GridSearchCV
    print("\nPerforming Grid Search for best parameters...")
    print("This will try 1 parameter combination with 2-fold cross-validation")
    grid_search.fit(X_train_tfidf, y_train)
    
    # Get the best model
    model = grid_search.best_estimator_
    print(f"\nBest parameters: {grid_search.best_params_}")
    
    # Make predictions
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Convert predictions back to original labels for reporting
    y_test_original = label_encoder.inverse_transform(y_test)
    y_pred_original = label_encoder.inverse_transform(y_pred)
    
    print(f"\nXGBoost Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test_original, y_pred_original))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test_original, y_pred_original)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('XGBoost - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('xgboost_confusion_matrix.png')
    print("\nConfusion matrix saved as 'xgboost_confusion_matrix.png'")
    
    # Plot feature importance
    feature_importance = model.feature_importances_
    feature_names = vectorizer.get_feature_names_out()
    
    # Create DataFrame for feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    })
    importance_df = importance_df.sort_values('importance', ascending=False).head(20)
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title('XGBoost - Top 20 Feature Importance')
    plt.tight_layout()
    plt.savefig('xgboost_feature_importance.png')
    print("\nFeature importance plot saved as 'xgboost_feature_importance.png'")
    
    # Save the model, vectorizer, and label encoder
    print("\nSaving model, vectorizer, and label encoder...")
    os.makedirs("models", exist_ok=True)
    with open("models/xgboost_model.pkl", "wb") as f:
        pickle.dump((vectorizer, model, label_encoder), f)
    
    # Test with example emails
    print("\nTesting with example emails:")
    test_emails = [
        "Dear customer, your account has been suspended. Please click here to verify your details.",
        "Meeting scheduled for tomorrow at 2 PM in Conference Room A.",
        "Your order #12345 has been shipped and will arrive in 2-3 business days."
    ]
    
    for email in test_emails:
        test_single_email(email, vectorizer, model, label_encoder)
    
    # Interactive testing
    while True:
        print("\n" + "="*50)
        print("Enter your email to test (or 'quit' to exit):")
        user_email = input("> ")
        
        if user_email.lower() == 'quit':
            break
            
        if user_email.strip():
            test_single_email(user_email, vectorizer, model, label_encoder)
        else:
            print("Please enter a valid email.")

if __name__ == "__main__":
    train_and_evaluate() 