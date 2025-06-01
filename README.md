# Email Classification System

This project implements an email classification system for a company's support team. The system categorizes incoming support emails into predefined categories while ensuring that personal information (PII) is masked before processing.

## Features

- PII/PCI Masking
- Email Classification
- REST API Endpoint
- Hugging Face Spaces Deployment

## Project Structure

```
email_classification/
├── api/
│   └── main.py           # FastAPI application
├── data/                 # Dataset storage
├── models/              # Trained models
├── utils/               # Utility functions
│   ├── masking.py       # PII masking logic
│   └── demasking.py     # PII demasking logic
├── notebook/            # Jupyter notebooks
├── classify.py          # Classification logic
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
```

## Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd email_classification
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## API Usage

The API is deployed at: `https://<username>-<space-name>.hf.space/classify`

### Endpoint: POST /classify

Input format:
```json
{
    "input_email_body": "string containing the email"
}
```

Output format:
```json
{
    "input_email_body": "string containing the email",
    "list_of_masked_entities": [
        {
            "position": [start_index, end_index],
            "classification": "entity_type",
            "entity": "original_entity_value"
        }
    ],
    "masked_email": "string containing the masked email",
    "category_of_the_email": "string containing the class"
}
```

## Categories

The system classifies emails into the following categories:
- Incident
- Request
- Change
- Problem

## PII/PCI Masking

The system masks the following types of personal information:
- Full Name
- Email Address
- Phone Number
- Date of Birth
- Aadhar Card Number
- Credit/Debit Card Number
- CVV Number
- Card Expiry Number 