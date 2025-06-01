import re
import spacy
from typing import List, Dict, Tuple

class PIIMasker:
    def __init__(self):
        # Load spaCy model
        self.nlp = spacy.load("en_core_web_sm")
        
        # Define regex patterns for different PII types
        self.patterns = {
            'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            'phone_number': r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            'aadhar_num': r'\d{4}[-.\s]?\d{4}[-.\s]?\d{4}',
            'credit_debit_no': r'\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}',
            'cvv_no': r'\b\d{3,4}\b',
            'expiry_no': r'\b(0[1-9]|1[0-2])[/-]\d{2}\b',
            'dob': r'\b(0[1-9]|1[0-2])[/-](0[1-9]|[12]\d|3[01])[/-]\d{4}\b'
        }

    def mask_text(self, text: str) -> Tuple[str, List[Dict]]:
        """
        Mask PII in the given text and return both masked text and list of masked entities.
        
        Args:
            text (str): Input text containing potential PII
            
        Returns:
            Tuple[str, List[Dict]]: Masked text and list of masked entities with their positions
        """
        masked_text = text
        masked_entities = []
        
        # Process with spaCy for name detection
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                masked_entities.append({
                    'position': [ent.start_char, ent.end_char],
                    'classification': 'full_name',
                    'entity': ent.text
                })
                masked_text = masked_text[:ent.start_char] + '[full_name]' + masked_text[ent.end_char:]
        
        # Process with regex patterns
        for entity_type, pattern in self.patterns.items():
            for match in re.finditer(pattern, masked_text):
                masked_entities.append({
                    'position': [match.start(), match.end()],
                    'classification': entity_type,
                    'entity': match.group()
                })
                masked_text = masked_text[:match.start()] + f'[{entity_type}]' + masked_text[match.end():]
        
        # Sort masked entities by their start position
        masked_entities.sort(key=lambda x: x['position'][0])
        
        return masked_text, masked_entities

    def validate_masking(self, original_text: str, masked_text: str, masked_entities: List[Dict]) -> bool:
        """
        Validate that masking was done correctly by checking if all masked entities
        can be restored to their original form.
        
        Args:
            original_text (str): Original text before masking
            masked_text (str): Text after masking
            masked_entities (List[Dict]): List of masked entities
            
        Returns:
            bool: True if masking is valid, False otherwise
        """
        # Create a copy of the masked text for validation
        validation_text = masked_text
        
        # Try to restore each masked entity
        for entity in reversed(masked_entities):  # Process in reverse to maintain positions
            start, end = entity['position']
            entity_type = entity['classification']
            original_entity = entity['entity']
            
            # Find the masked placeholder
            placeholder = f'[{entity_type}]'
            placeholder_pos = validation_text.find(placeholder)
            
            if placeholder_pos == -1:
                return False
                
            # Replace the placeholder with the original entity
            validation_text = (
                validation_text[:placeholder_pos] +
                original_entity +
                validation_text[placeholder_pos + len(placeholder):]
            )
        
        # Compare with original text
        return validation_text == original_text 