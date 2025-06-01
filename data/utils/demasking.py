from typing import List, Dict

class PIIDemasker:
    def __init__(self):
        pass

    def demask_text(self, masked_text: str, masked_entities: List[Dict]) -> str:
        """
        Restore the original text by replacing masked entities with their original values.
        
        Args:
            masked_text (str): Text containing masked entities
            masked_entities (List[Dict]): List of masked entities with their positions and original values
            
        Returns:
            str: Original text with all masked entities restored
        """
        # Sort entities by their start position in reverse order
        # This ensures we don't mess up positions when replacing
        sorted_entities = sorted(masked_entities, key=lambda x: x['position'][0], reverse=True)
        
        demasked_text = masked_text
        
        for entity in sorted_entities:
            entity_type = entity['classification']
            original_value = entity['entity']
            placeholder = f'[{entity_type}]'
            
            # Replace the placeholder with the original value
            demasked_text = demasked_text.replace(placeholder, original_value, 1)
        
        return demasked_text

    def validate_demasking(self, original_text: str, demasked_text: str) -> bool:
        """
        Validate that demasking was done correctly by comparing with the original text.
        
        Args:
            original_text (str): Original text before masking
            demasked_text (str): Text after demasking
            
        Returns:
            bool: True if demasking is valid, False otherwise
        """
        return original_text == demasked_text 