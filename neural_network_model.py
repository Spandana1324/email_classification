import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
import time
from utils.masking import PIIMasker
from utils.demasking import PIIDemasker
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, Conv1D, MaxPooling1D, Input, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

def preprocess_data(df, max_samples=None):
    """Preprocess the email data"""
    print("\nPreprocessing Data:")
    print(f"Total samples in dataset: {len(df)}")
    
    # If max_samples is specified, take a random subset
    if max_samples and max_samples < len(df):
        df = df.sample(n=max_samples, random_state=42)
        print(f"Using {max_samples} random samples for processing")
    
    # Initialize masker and demasker
    masker = PIIMasker()
    demasker = PIIDemasker()
    
    # Remove any missing values
    df = df.dropna()
    print(f"\nSamples after removing missing values: {len(df)}")
    
    # Apply masking to email content (process in batches)
    print("\nApplying PII masking to emails...")
    batch_size = 500  # Increased batch size for faster processing
    masked_emails = []
    masked_entities_list = []
    
    for i in range(0, len(df), batch_size):
        batch = df['email'].iloc[i:i+batch_size]
        for email in batch:
            masked_text, masked_entities = masker.mask_text(email)
            masked_emails.append(masked_text)
            masked_entities_list.append(masked_entities)
        print(f"Processed {min(i+batch_size, len(df))} emails...")
    
    # Add masked emails to dataframe
    df['masked_email'] = masked_emails
    df['masked_entities'] = masked_entities_list
    
    return df

def create_cnn_model(vocab_size, max_length, num_classes, embedding_dim=200):
    """Create an enhanced CNN model for better feature extraction"""
    inputs = Input(shape=(max_length,))
    
    # Enhanced embedding layer with larger dimension
    x = Embedding(vocab_size, embedding_dim, input_length=max_length)(inputs)
    
    # Multiple CNN layers with different filter sizes
    conv1 = Conv1D(256, 5, activation='relu', padding='same')(x)
    conv2 = Conv1D(256, 3, activation='relu', padding='same')(x)
    conv3 = Conv1D(256, 7, activation='relu', padding='same')(x)
    
    # Concatenate different filter sizes
    x = tf.keras.layers.concatenate([conv1, conv2, conv3])
    x = MaxPooling1D(2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Second CNN block
    conv4 = Conv1D(128, 5, activation='relu', padding='same')(x)
    conv5 = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.concatenate([conv4, conv5])
    x = MaxPooling1D(2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Third CNN block
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Dense layers with residual connections
    x = Flatten()(x)
    dense1 = Dense(256, activation='relu')(x)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.4)(dense1)
    
    dense2 = Dense(128, activation='relu')(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Dropout(0.4)(dense2)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(dense2)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model with better optimizer settings
    optimizer = Adam(learning_rate=0.0005)  # Reduced learning rate
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def test_single_email(email, tokenizer, cnn_model, rf_model, max_length, masker, demasker):
    """Test a single email with the hybrid model"""
    print("\n" + "="*80)
    print("TESTING EMAIL WITH MASKING/DEMASKING VERIFICATION")
    print("="*80)
    
    # Original email
    print("\n1. Original Email:")
    print("-"*40)
    print(email)
    
    # Apply masking
    masked_email, masked_entities = masker.mask_text(email)
    print("\n2. Masked Email:")
    print("-"*40)
    print(masked_email)
    
    # Show what was masked
    print("\n3. Masked Entities:")
    print("-"*40)
    for entity in masked_entities:
        print(f"Type: {entity['classification']}")
        print(f"Original: {entity['entity']}")
        print(f"Position: {entity['position']}")
        print("-"*20)
    
    # Apply demasking
    demasked_email = demasker.demask_text(masked_email, masked_entities)
    print("\n4. Demasked Email (should match original):")
    print("-"*40)
    print(demasked_email)
    
    # Verify demasking
    print("\n5. Verification:")
    print("-"*40)
    verification_passed = True
    for entity in masked_entities:
        original_value = entity['entity']
        if original_value not in demasked_email:
            verification_passed = False
            print(f"Entity '{original_value}' not found in demasked text")
    
    if verification_passed:
        print("✓ Demasking successful: All masked entities were properly restored")
    else:
        print("✗ Demasking failed: Some masked entities were not properly restored")
    
    # Make prediction using hybrid model
    print("\n6. Classification Results:")
    print("-"*40)
    
    # Preprocess the masked email
    sequence = tokenizer.texts_to_sequences([masked_email])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    
    # Extract features using CNN
    features = cnn_model.predict(padded_sequence)
    
    # Get prediction from Random Forest
    prediction = rf_model.predict_proba(features)
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]
    
    print(f"Predicted category: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    
    print("\n" + "="*80)

def train_and_evaluate(max_samples=1000):  # Increased sample size
    """Train and evaluate enhanced hybrid CNN-Random Forest model"""
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
    
    # Prepare text data
    texts = df['masked_email'].values
    labels = df['type'].values
    
    # Create and fit tokenizer with larger vocabulary
    print("\nCreating tokenizer...")
    tokenizer = Tokenizer(num_words=10000)  # Increased vocabulary size
    tokenizer.fit_on_texts(texts)
    
    # Convert texts to sequences
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Pad sequences
    max_length = 150  # Increased sequence length
    X = pad_sequences(sequences, maxlen=max_length, padding='post')
    
    # Convert labels to numeric
    label_encoder = {label: i for i, label in enumerate(np.unique(labels))}
    y = np.array([label_encoder[label] for label in labels])
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Create and train CNN model
    print("\nTraining CNN feature extractor...")
    cnn_start = time.time()
    cnn_model = create_cnn_model(
        vocab_size=len(tokenizer.word_index) + 1,
        max_length=max_length,
        num_classes=len(label_encoder)
    )
    
    # Enhanced training with learning rate scheduling
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=0.00001
    )
    
    # Train CNN with enhanced callbacks
    cnn_model.fit(
        X_train, y_train,
        epochs=20,  # Increased epochs
        batch_size=32,
        validation_split=0.1,
        callbacks=[
            EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            reduce_lr
        ],
        verbose=1
    )
    print(f"CNN training time: {time.time() - cnn_start:.2f} seconds")
    
    # Extract features using CNN
    print("\nExtracting features...")
    feature_start = time.time()
    # Get features from the second-to-last layer
    feature_extractor = Model(inputs=cnn_model.input, 
                            outputs=cnn_model.layers[-2].output)
    train_features = feature_extractor.predict(X_train)
    test_features = feature_extractor.predict(X_test)
    print(f"Feature extraction time: {time.time() - feature_start:.2f} seconds")
    
    # Train Random Forest with enhanced parameters
    print("\nTraining Random Forest classifier...")
    rf_start = time.time()
    rf_model = RandomForestClassifier(
        n_estimators=300,  # Increased number of trees
        max_depth=20,      # Increased depth
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    
    rf_model.fit(train_features, y_train)
    print(f"Random Forest training time: {time.time() - rf_start:.2f} seconds")
    
    # Evaluate model
    print("\nEvaluating model...")
    eval_start = time.time()
    y_pred = rf_model.predict(test_features)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Evaluation time: {time.time() - eval_start:.2f} seconds")
    
    # Save models and tokenizer
    print("\nSaving models and tokenizer...")
    save_start = time.time()
    os.makedirs("models", exist_ok=True)
    cnn_model.save("models/cnn_feature_extractor.keras")
    with open("models/random_forest.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    with open("models/tokenizer.pkl", "wb") as f:
        pickle.dump((tokenizer, label_encoder, max_length, masker, demasker), f)
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
        test_single_email(email, tokenizer, feature_extractor, rf_model, max_length, masker, demasker)
    
    # Interactive testing
    while True:
        print("\n" + "="*80)
        print("Enter your email to test (or 'quit' to exit):")
        user_email = input("> ")
        
        if user_email.lower() == 'quit':
            break
            
        if user_email.strip():
            test_single_email(user_email, tokenizer, feature_extractor, rf_model, max_length, masker, demasker)
        else:
            print("Please enter a valid email.")

if __name__ == "__main__":
    train_and_evaluate(max_samples=1000)  # Process 1000 emails by default 