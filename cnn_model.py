import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer
import pickle
import os

class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class EmailCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, num_filters, filter_sizes):
        super(EmailCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim)) 
            for k in filter_sizes
        ])
        
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        embedded = embedded.unsqueeze(1)  # [batch_size, 1, seq_len, embedding_dim]
        
        # Apply convolutions and pooling
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(embedded)).squeeze(3)
            pool_out = torch.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pool_out)
        
        # Concatenate and pass through final layer
        concat_out = torch.cat(conv_outputs, dim=1)
        concat_out = self.dropout(concat_out)
        return self.fc(concat_out)

def train_and_evaluate():
    print("\nLoading dataset...")
    df = pd.read_csv("data/email_dataset.csv")
    
    # Preprocess data
    df = df.dropna()
    print(f"\nTotal samples: {len(df)}")
    print("\nClass distribution:")
    print(df['type'].value_counts())
    
    # Split features and target
    X = df['email'].values
    y = df['type'].values
    
    # Convert labels to numerical values
    label_encoder = {label: idx for idx, label in enumerate(np.unique(y))}
    y = np.array([label_encoder[label] for label in y])
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create datasets
    train_dataset = EmailDataset(X_train, y_train, tokenizer)
    test_dataset = EmailDataset(X_test, y_test, tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EmailCNN(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=300,
        num_classes=len(label_encoder),
        num_filters=100,
        filter_sizes=[3, 4, 5]
    ).to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print("\nTraining CNN model...")
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
    
    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label']
            
            outputs = model(input_ids)
            _, preds = torch.max(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nCNN Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('CNN - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('cnn_confusion_matrix.png')
    print("\nConfusion matrix saved as 'cnn_confusion_matrix.png'")
    
    # Save the model and label encoder
    print("\nSaving model and label encoder...")
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/cnn_model.pth")
    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

def test_single_email(email, model, tokenizer, label_encoder, device):
    """Test a single email and show detailed prediction results"""
    print("\n" + "="*50)
    print("Testing Email:")
    print("-"*50)
    print(f"Email content: {email}")
    print("-"*50)
    
    # Prepare input
    encoding = tokenizer.encode_plus(
        email,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
        probabilities = torch.softmax(outputs, dim=1)[0]
        prediction = torch.argmax(probabilities).item()
    
    # Convert prediction back to original label
    reverse_label_encoder = {v: k for k, v in label_encoder.items()}
    predicted_label = reverse_label_encoder[prediction]
    
    # Print results
    print("\nPrediction Results:")
    print("-"*50)
    print(f"Predicted category: {predicted_label}")
    print("\nConfidence scores for each category:")
    for label, idx in label_encoder.items():
        print(f"{label}: {probabilities[idx].item():.2%}")
    print("="*50)

if __name__ == "__main__":
    train_and_evaluate() 