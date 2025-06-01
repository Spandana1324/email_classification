import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import pickle
import os
import time
from utils.masking import PIIMasker
from utils.demasking import PIIDemasker
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from collections import Counter
import spacy

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except:
    import subprocess
    subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
    nlp = spacy.load('en_core_web_sm')

def clean_text(text):
    """Enhanced text cleaning with lemmatization and stop words removal"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs and email addresses
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^a-z0-9\s.,!?]', '', text)
    
    # Process with spaCy
    doc = nlp(text)
    
    # Lemmatize and remove stop words
    words = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    
    # Join back into text
    text = ' '.join(words)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def extract_features(text):
    """Extract advanced features from text"""
    features = {}
    
    # Basic text features
    features['length'] = len(text)
    features['word_count'] = len(text.split())
    features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
    
    # Special character counts
    features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
    features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text) if len(text) > 0 else 0
    features['special_char_ratio'] = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text) if len(text) > 0 else 0
    
    # Process with spaCy
    doc = nlp(text)
    
    # Named Entity features
    features['ner_count'] = len(doc.ents)
    features['person_count'] = len([ent for ent in doc.ents if ent.label_ == 'PERSON'])
    features['org_count'] = len([ent for ent in doc.ents if ent.label_ == 'ORG'])
    features['date_count'] = len([ent for ent in doc.ents if ent.label_ == 'DATE'])
    
    # POS tag features
    pos_counts = Counter([token.pos_ for token in doc])
    features['noun_ratio'] = pos_counts.get('NOUN', 0) / len(doc) if len(doc) > 0 else 0
    features['verb_ratio'] = pos_counts.get('VERB', 0) / len(doc) if len(doc) > 0 else 0
    features['adj_ratio'] = pos_counts.get('ADJ', 0) / len(doc) if len(doc) > 0 else 0
    
    # Dependency features - using numeric values
    dep_counts = Counter([token.dep_ for token in doc])
    features['nsubj_count'] = dep_counts.get('nsubj', 0) / len(doc) if len(doc) > 0 else 0
    features['dobj_count'] = dep_counts.get('dobj', 0) / len(doc) if len(doc) > 0 else 0
    features['prep_count'] = dep_counts.get('prep', 0) / len(doc) if len(doc) > 0 else 0
    features['pobj_count'] = dep_counts.get('pobj', 0) / len(doc) if len(doc) > 0 else 0
    
    # Category-specific keywords
    category_keywords = {
        'incident': ['incident', 'issue', 'error', 'failed', 'down', 'broken', 'outage', 'urgent', 'critical'],
        'change': ['change', 'update', 'modify', 'implement', 'deploy', 'release', 'version', 'upgrade'],
        'problem': ['problem', 'trouble', 'difficulty', 'concern', 'issue', 'bug', 'defect', 'fault'],
        'meeting': ['meeting', 'schedule', 'agenda', 'attend', 'discuss', 'presentation', 'conference', 'call']
    }
    
    for category, keywords in category_keywords.items():
        features[f'{category}_keyword_count'] = sum(1 for word in text.lower().split() if word in keywords)
        features[f'{category}_keyword_ratio'] = features[f'{category}_keyword_count'] / len(text.split()) if text.split() else 0
    
    return features

def preprocess_data(df, max_samples=None):
    """Enhanced preprocessing with additional features"""
    print("\nPreprocessing Data:")
    print(f"Total samples in dataset: {len(df)}")
    
    if max_samples and max_samples < len(df):
        df = df.sample(n=max_samples, random_state=42)
        print(f"Using {max_samples} random samples for processing")
    
    # Initialize masker and demasker
    masker = PIIMasker()
    demasker = PIIDemasker()
    
    # Remove missing values
    df = df.dropna()
    print(f"\nSamples after removing missing values: {len(df)}")
    
    # Clean and mask emails
    print("\nCleaning and masking emails...")
    batch_size = 1000
    masked_emails = []
    masked_entities_list = []
    cleaned_emails = []
    additional_features = []
    
    for i in range(0, len(df), batch_size):
        batch = df['email'].iloc[i:i+batch_size]
        for email in batch:
            # Clean text
            cleaned_text = clean_text(email)
            cleaned_emails.append(cleaned_text)
            
            # Extract additional features
            features = extract_features(cleaned_text)
            additional_features.append(features)
            
            # Apply masking
            masked_text, masked_entities = masker.mask_text(email)
            masked_emails.append(masked_text)
            masked_entities_list.append(masked_entities)
        print(f"Processed {min(i+batch_size, len(df))} emails...")
    
    # Add processed data to dataframe
    df['cleaned_email'] = cleaned_emails
    df['masked_email'] = masked_emails
    df['masked_entities'] = masked_entities_list
    df['additional_features'] = additional_features
    
    return df

def test_single_email(email, vectorizer, svm_model, masker, demasker):
    """Test a single email with classification"""
    print("\n" + "="*80)
    print("CLASSIFICATION RESULTS")
    print("="*80)
    
    # Apply masking
    masked_email, masked_entities = masker.mask_text(email)
    
    # Apply demasking
    demasked_email = demasker.demask_text(masked_email, masked_entities)
    
    # Clean and prepare text
    cleaned_text = clean_text(masked_email)
    
    # Transform the masked email
    text_features = vectorizer.transform([masked_email])
    
    # Extract additional features
    additional_features = extract_features(cleaned_text)
    feature_df = pd.DataFrame([additional_features])
    
    # Combine text and additional features
    from scipy.sparse import hstack
    features = hstack([text_features, feature_df])
    
    # Get prediction and probabilities
    prediction = svm_model.predict(features)[0]
    probabilities = svm_model.predict_proba(features)[0]
    
    # Get the confidence for the predicted class
    confidence = probabilities[np.argmax(probabilities)]
    
    print(f"Predicted category: {prediction}")
    print(f"Confidence: {confidence:.2%}")  # Format as percentage
    
    # Show top 3 predictions if available
    if len(probabilities) > 1:
        print("\nTop 3 predictions:")
        top_indices = np.argsort(probabilities)[-3:][::-1]
        for idx in top_indices:
            class_name = svm_model.classes_[idx]
            prob = probabilities[idx]
            print(f"- {class_name}: {prob:.2%}")
    
    print("\n" + "="*80)

def train_and_evaluate(max_samples=1000):
    """Train and evaluate enhanced ensemble model"""
    start_time = time.time()
    print("\nLoading dataset...")
    df = pd.read_csv("data/email_dataset.csv")
    
    # Initialize masker and demasker
    masker = PIIMasker()
    demasker = PIIDemasker()
    
    # Preprocess data
    print("\nPreprocessing data...")
    preprocess_start = time.time()
    df = preprocess_data(df, max_samples)
    print(f"Preprocessing time: {time.time() - preprocess_start:.2f} seconds")
    
    # Prepare text data and additional features
    texts = df['masked_email'].values
    labels = df['type'].values
    
    # Create TF-IDF vectorizer with optimized parameters
    print("\nCreating TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=20000,  # Increased features
        ngram_range=(1, 4),  # Added 4-grams
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        use_idf=True,
        smooth_idf=True
    )
    
    # Transform texts
    X_text = vectorizer.fit_transform(texts)
    
    # Extract additional features
    print("\nExtracting additional features...")
    feature_df = pd.DataFrame(df['additional_features'].tolist())
    X_features = feature_df.values
    
    # Combine text and additional features
    from scipy.sparse import hstack
    X = hstack([X_text, X_features])
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}")
    
    # Create base models
    svm = SVC(
        kernel='rbf',
        probability=True,
        class_weight='balanced',
        cache_size=2000,
        random_state=42
    )
    
    lr = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    
    nb = MultinomialNB()
    
    # Create ensemble model
    ensemble = VotingClassifier(
        estimators=[
            ('svm', svm),
            ('lr', lr),
            ('nb', nb)
        ],
        voting='soft',
        weights=[2, 1, 1]  # Give more weight to SVM
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('ensemble', ensemble)
    ])
    
    # Define parameter grid for GridSearchCV
    param_grid = {
        'ensemble__svm__C': [1, 10, 100],
        'ensemble__svm__gamma': ['scale', 'auto', 0.1],
        'ensemble__lr__C': [0.1, 1, 10],
        'ensemble__weights': [[2, 1, 1], [3, 1, 1], [2, 2, 1]]
    }
    
    # Perform GridSearchCV
    print("\nPerforming hyperparameter tuning...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Train model
    print("\nTraining model...")
    train_start = time.time()
    grid_search.fit(X_train, y_train)
    print(f"Training time: {time.time() - train_start:.2f} seconds")
    
    # Get best model
    best_model = grid_search.best_estimator_
    print("\nBest parameters:", grid_search.best_params_)
    
    # Evaluate model
    print("\nEvaluating model...")
    eval_start = time.time()
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Evaluation time: {time.time() - eval_start:.2f} seconds")
    
    # Save models and vectorizer
    print("\nSaving models and vectorizer...")
    save_start = time.time()
    os.makedirs("models", exist_ok=True)
    with open("models/ensemble_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open("models/vectorizer.pkl", "wb") as f:
        pickle.dump((vectorizer, masker, demasker), f)
    print(f"Model saving time: {time.time() - save_start:.2f} seconds")
    
    total_time = time.time() - start_time
    print(f"\nTotal training and evaluation time: {total_time:.2f} seconds")
    
    # Test with example emails
    print("\nTesting with example emails:")
    test_emails = [
        "Dear John Smith, your account has been suspended. Please contact us at support@example.com or call +1-555-123-4567.",
        "Meeting with Sarah Johnson scheduled for 12/25/2023 at 2 PM in Conference Room A.",
        "Your order #12345 has been shipped to 123 Main St, New York, NY 10001. CVV: 123, Expiry: 12/25."
    ]
    
    for email in test_emails:
        test_single_email(email, vectorizer, best_model, masker, demasker)
    
    # Interactive testing
    while True:
        print("\n" + "="*80)
        print("Enter your email to test (or 'quit' to exit):")
        user_email = input("> ")
        
        if user_email.lower() == 'quit':
            break
            
        if user_email.strip():
            test_single_email(user_email, vectorizer, best_model, masker, demasker)
        else:
            print("Please enter a valid email.")

if __name__ == "__main__":
    train_and_evaluate(max_samples=1000)  # Process 1000 emails by default 