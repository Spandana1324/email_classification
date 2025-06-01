import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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

def extract_advanced_features(text):
    """Extract advanced features specific to email classification"""
    features = {}
    
    # Basic text features
    features['length'] = len(text)
    features['word_count'] = len(text.split())
    features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
    
    # Special character counts
    features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
    features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text) if len(text) > 0 else 0
    features['special_char_ratio'] = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text) if len(text) > 0 else 0
    
    # Word frequency features
    words = text.lower().split()
    word_freq = Counter(words)
    features['unique_word_ratio'] = len(word_freq) / len(words) if words else 0
    features['most_common_word_freq'] = word_freq.most_common(1)[0][1] / len(words) if words else 0
    
    # Category-specific features
    category_keywords = {
        'incident': ['incident', 'issue', 'error', 'failed', 'down', 'broken', 'outage', 'urgent', 'critical'],
        'change': ['change', 'update', 'modify', 'implement', 'deploy', 'release', 'version', 'upgrade'],
        'problem': ['problem', 'trouble', 'difficulty', 'concern', 'issue', 'bug', 'defect', 'fault'],
        'meeting': ['meeting', 'schedule', 'agenda', 'attend', 'discuss', 'presentation', 'conference', 'call']
    }
    
    for category, keywords in category_keywords.items():
        features[f'{category}_keyword_count'] = sum(1 for word in words if word in keywords)
        features[f'{category}_keyword_ratio'] = features[f'{category}_keyword_count'] / len(words) if words else 0
    
    # Time-related features
    time_indicators = ['today', 'tomorrow', 'yesterday', 'week', 'month', 'year', 'date', 'time', 'schedule']
    features['time_indicator_count'] = sum(1 for word in words if word in time_indicators)
    
    # Urgency indicators
    urgency_indicators = ['urgent', 'asap', 'immediate', 'critical', 'important', 'priority']
    features['urgency_indicator_count'] = sum(1 for word in words if word in urgency_indicators)
    
    return features

def clean_text(text):
    """Advanced text cleaning with lemmatization"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs and email addresses
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^a-z0-9\s.,!?]', '', text)
    
    # Tokenize and lemmatize
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    # Join back into text
    text = ' '.join(words)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def train_optimized_model():
    """Train an optimized model with advanced features and select the best"""
    print("Loading dataset...")
    df = pd.read_csv("data/email_dataset.csv")
    
    # Clean the data
    print("Cleaning data...")
    df['cleaned_email'] = df['email'].apply(clean_text)
    
    # Extract advanced features
    print("Extracting advanced features...")
    feature_list = []
    for text in df['cleaned_email']:
        features = extract_advanced_features(text)
        feature_list.append(features)
    
    # Convert features to DataFrame
    feature_df = pd.DataFrame(feature_list)
    
    # Remove empty emails after cleaning
    df = df[df['cleaned_email'].str.len() > 0]
    feature_df = feature_df[df['cleaned_email'].str.len() > 0]
    
    # Balance classes
    print("Balancing classes...")
    class_counts = df['type'].value_counts()
    min_count = class_counts.min()
    balanced_indices = []
    for cls in class_counts.index:
        class_indices = df[df['type'] == cls].index
        balanced_indices.extend(np.random.choice(class_indices, size=min_count, replace=False))
    df = df.loc[balanced_indices]
    feature_df = feature_df.loc[balanced_indices]
    
    # Split data
    X_text = df['cleaned_email'].values
    X_features = feature_df.values
    y = df['type'].values
    
    # Split into train and test sets
    X_text_train, X_text_test, X_features_train, X_features_test, y_train, y_test = train_test_split(
        X_text, X_features, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create optimized vectorizer
    print("Training vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=50000,     # Increased features
        min_df=2,              # Minimum document frequency
        max_df=0.90,           # Maximum document frequency
        ngram_range=(1, 3),    # Using 1-3 grams
        sublinear_tf=True,     # Apply sublinear scaling
        use_idf=True,          # Use IDF weighting
        smooth_idf=True        # Smooth IDF weights
    )
    
    # Transform text data
    X_text_train_tfidf = vectorizer.fit_transform(X_text_train)
    X_text_test_tfidf = vectorizer.transform(X_text_test)
    
    # Combine text and features
    X_train_combined = np.hstack([X_text_train_tfidf.toarray(), X_features_train])
    X_test_combined = np.hstack([X_text_test_tfidf.toarray(), X_features_test])
    
    # Define multiple models to evaluate
    print("Evaluating multiple models...")
    models = {
        'rf': RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'svm': SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42)
    }
    
    best_accuracy = 0
    best_model = None
    best_model_name = ""
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_combined, y_train)
        y_pred = model.predict(X_test_combined)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} Test Set Accuracy: {accuracy:.2f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name
    
    print(f"\nBest performing model: {best_model_name} with Accuracy: {best_accuracy:.2f}")
    print("\nDetailed Classification Report for Best Model:")
    print(classification_report(y_test, best_model.predict(X_test_combined)))
    
    # Save the best model and vectorizer
    print("\nSaving best model...")
    with open('optimized_model.pkl', 'wb') as f:
        pickle.dump((vectorizer, best_model, feature_df.columns), f)
    print("Best model saved as 'optimized_model.pkl'")
    
    return vectorizer, best_model, feature_df.columns

def predict_email(email_text, vectorizer, model, feature_columns):
    """Make optimized predictions on new emails"""
    # Clean the email text
    cleaned_text = clean_text(email_text)
    
    # Extract features
    features = extract_advanced_features(cleaned_text)
    feature_array = np.array([features.get(col, 0) for col in feature_columns]).reshape(1, -1)
    
    # Transform the text
    text_tfidf = vectorizer.transform([cleaned_text])
    
    # Combine text and features
    combined_features = np.hstack([text_tfidf.toarray(), feature_array])
    
    # Get prediction and probabilities
    prediction = model.predict(combined_features)[0]
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(combined_features)[0]
        confidence = probabilities.max()
    else:
        confidence = 1.0
        print("Warning: Model does not support probability estimates. Confidence set to 1.0.")
    
    return prediction, confidence

def real_time_prediction(vectorizer, model, feature_columns):
    """Handle real-time email predictions"""
    print("\n=== Real-time Email Classification ===")
    print("Type 'quit' to exit")
    print("Enter your email text below:")
    
    while True:
        print("\n" + "="*50)
        email = input("\nEnter email text: ").strip()
        
        if email.lower() == 'quit':
            print("\nExiting real-time prediction mode...")
            break
            
        if not email:
            print("Please enter some text!")
            continue
            
        try:
            prediction, confidence = predict_email(email, vectorizer, model, feature_columns)
            print("\nResults:")
            print(f"Classification: {prediction}")
            print(f"Confidence: {confidence:.2%}")
            
            # Add detailed interpretation
            if confidence > 0.9:
                print("High confidence prediction - Very likely correct")
            elif confidence > 0.7:
                print("Moderate confidence prediction - Likely correct")
            elif confidence > 0.5:
                print("Low confidence prediction - Consider reviewing manually")
            else:
                print("Very low confidence prediction - Manual review recommended")
                
        except Exception as e:
            print(f"Error processing email: {str(e)}")
            print("Please try again with different text")

if __name__ == "__main__":
    # Train the optimized model and select the best
    print("Training and evaluating models (this may take a few minutes)...")
    vectorizer, model, feature_columns = train_optimized_model()
    
    # Start real-time prediction with the best model
    real_time_prediction(vectorizer, model, feature_columns) 