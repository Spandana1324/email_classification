import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import os
import sys
from tqdm import tqdm

def download_nltk_data():
    """Download required NLTK data with progress tracking"""
    print("Downloading required NLTK data...")
    resources = ['stopwords', 'wordnet', 'punkt', 'averaged_perceptron_tagger']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
            print(f"✓ Downloaded {resource}")
        except Exception as e:
            print(f"✗ Error downloading {resource}: {str(e)}")
    print("NLTK data download completed.\n")

def advanced_preprocess_text(text):
    """Advanced text preprocessing"""
    if not isinstance(text, str):
        return ""
    
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenization
        tokens = nltk.word_tokenize(text)
        
        # Remove stopwords and lemmatize
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        
        return ' '.join(tokens)
    except Exception as e:
        print(f"Error in preprocessing text: {str(e)}")
        return text  # Return original text if preprocessing fails

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
    
    # Apply advanced preprocessing to email content with progress bar
    print("\nApplying advanced text preprocessing...")
    tqdm.pandas(desc="Preprocessing emails")
    df['processed_email'] = df['email'].progress_apply(advanced_preprocess_text)
    
    # Check class distribution
    print("\nClass distribution:")
    print(df['type'].value_counts())
    
    return df

def train_and_evaluate():
    """Train and evaluate optimized Naive Bayes model"""
    print("\nLoading dataset...")
    df = pd.read_csv("data/email_dataset.csv")
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Split features and target
    X = df['processed_email'].values
    y = df['type'].values
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Create pipeline with TF-IDF and Naive Bayes
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 3),
            sublinear_tf=True
        )),
        ('nb', MultinomialNB())
    ])
    
    # Define parameter grid for GridSearchCV
    param_grid = {
        'tfidf__max_features': [5000, 10000, 15000],
        'tfidf__ngram_range': [(1, 2), (1, 3)],
        'tfidf__min_df': [2, 3, 5],
        'nb__alpha': [0.1, 0.5, 1.0, 2.0]
    }
    
    # Perform GridSearchCV
    print("\nPerforming GridSearchCV for hyperparameter tuning...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        n_jobs=-1,
        scoring='accuracy',
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    print(f"\nBest parameters: {grid_search.best_params_}")
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nOptimized Naive Bayes Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Optimized Naive Bayes - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('optimized_nb_confusion_matrix.png')
    print("\nConfusion matrix saved as 'optimized_nb_confusion_matrix.png'")
    
    # Save the best model
    print("\nSaving the best model...")
    with open('optimized_nb_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print("Model saved as 'optimized_nb_model.pkl'")
    
    return best_model

if __name__ == "__main__":
    # Download NLTK data first
    download_nltk_data()
    
    # Train and evaluate the model
    train_and_evaluate() 