import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
from utils.masking import PIIMasker
from utils.demasking import PIIDemasker
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, Bidirectional, Input, BatchNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

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
    batch_size = 1000  # Increased batch size for faster processing
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

def create_lstm_model(vocab_size, max_length, num_classes):
    """Create an optimized Bidirectional LSTM model for faster training"""
    # Input layer
    inputs = Input(shape=(max_length,))
    
    # Optimized embedding layer
    embedding = Embedding(vocab_size, 128, input_length=max_length)(inputs)
    
    # Single Bidirectional LSTM layer with regularization
    lstm = Bidirectional(LSTM(64, return_sequences=False,
                            kernel_regularizer=l2(1e-4)))(embedding)
    lstm = BatchNormalization()(lstm)
    lstm = Dropout(0.3)(lstm)
    
    # Optimized dense layers
    dense = Dense(32, activation='relu', kernel_regularizer=l2(1e-4))(lstm)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.3)(dense)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(dense)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model with optimized settings
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def test_single_email(email, tokenizer, model, max_length, masker, demasker):
    """Test a single email and show prediction results with masking/demasking verification"""
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
    
    # Simple verification: Check if all masked entities are properly restored
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
    
    # Make prediction using masked email
    print("\n6. Classification Results:")
    print("-"*40)
    
    # Preprocess the masked email
    sequence = tokenizer.texts_to_sequences([masked_email])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    
    # Get prediction
    prediction = model.predict(padded_sequence, verbose=0)
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]
    
    print(f"Predicted category: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    
    print("\n" + "="*80)

def tune_hyperparameters(X_train, y_train, vocab_size, max_length, num_classes):
    """Perform hyperparameter tuning using GridSearchCV"""
    print("\nPerforming hyperparameter tuning...")
    
    # Create model wrapper for scikit-learn
    model = KerasClassifier(
        build_fn=lambda: create_lstm_model(vocab_size, max_length, num_classes),
        epochs=5,
        batch_size=64,
        verbose=0
    )
    
    # Enhanced hyperparameter grid
    param_grid = {
        'lstm_units': [64, 128],  # Increased LSTM units
        'dropout_rate': [0.2, 0.3, 0.4],  # Adjusted dropout rates
        'embedding_dim': [128, 256],  # Increased embedding dimensions
        'batch_size': [32, 64, 128],  # More batch size options
        'epochs': [5, 10]  # Increased epochs
    }
    
    # Create GridSearchCV object
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    
    # Perform grid search
    grid_result = grid.fit(X_train, y_train)
    
    # Print best parameters
    print("\nBest parameters found:")
    print(grid_result.best_params_)
    
    return grid_result.best_params_

def train_and_evaluate(max_samples=500):  # Reduced samples for faster training
    """Train and evaluate LSTM model with optimized settings"""
    print("\nLoading dataset...")
    df = pd.read_csv("data/email_dataset.csv")
    
    # Initialize masker and demasker
    masker = PIIMasker()
    demasker = PIIDemasker()
    
    # Preprocess data
    df = preprocess_data(df, max_samples)
    
    # Prepare text data
    texts = df['masked_email'].values
    labels = df['type'].values
    
    # Create and fit tokenizer with reduced vocabulary
    print("\nCreating tokenizer...")
    tokenizer = Tokenizer(num_words=5000)  # Reduced vocabulary size
    tokenizer.fit_on_texts(texts)
    
    # Convert texts to sequences
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Pad sequences with reduced length
    max_length = 100  # Reduced sequence length
    X = pad_sequences(sequences, maxlen=max_length, padding='post')
    
    # Convert labels to numeric
    label_encoder = {label: i for i, label in enumerate(np.unique(labels))}
    y = np.array([label_encoder[label] for label in labels])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Create and train model
    print("\nCreating and training LSTM model...")
    model = create_lstm_model(
        vocab_size=len(tokenizer.word_index) + 1,
        max_length=max_length,
        num_classes=len(label_encoder)
    )
    
    # Optimized callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=0.0001
        )
    ]
    
    # Train model with optimized settings
    history = model.fit(
        X_train, y_train,
        epochs=10,  # Reduced epochs
        batch_size=64,  # Increased batch size for faster training
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    y_pred = np.argmax(model.predict(X_test), axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model and tokenizer
    print("\nSaving model and tokenizer...")
    os.makedirs("models", exist_ok=True)
    model.save("models/lstm_model.keras")
    with open("models/tokenizer.pkl", "wb") as f:
        pickle.dump((tokenizer, label_encoder, max_length, masker, demasker), f)
    
    # Test with example emails
    print("\nTesting with example emails:")
    test_emails = [
        "Dear John Smith, your account has been suspended. Please contact us at support@example.com or call +1-555-123-4567.",
        "Meeting with Sarah Johnson scheduled for 12/25/2023 at 2 PM in Conference Room A.",
        "Your order #12345 has been shipped to 123 Main St, New York, NY 10001. CVV: 123, Expiry: 12/25."
    ]
    
    for email in test_emails:
        test_single_email(email, tokenizer, model, max_length, masker, demasker)
    
    # Interactive testing
    while True:
        print("\n" + "="*80)
        print("Enter your email to test (or 'quit' to exit):")
        user_email = input("> ")
        
        if user_email.lower() == 'quit':
            break
            
        if user_email.strip():
            test_single_email(user_email, tokenizer, model, max_length, masker, demasker)
        else:
            print("Please enter a valid email.")

if __name__ == "__main__":
    train_and_evaluate(max_samples=500)  # Process 500 emails by default 