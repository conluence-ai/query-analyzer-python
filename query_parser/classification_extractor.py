""" Classification Extractor for furniture queries """

# Import necessary libraries
import re
import logging
from typing import Dict, List, Optional

# Import custom modules
from utils.helpers import spellCorrect, fuzzyMatch

# Import constants and mappings
from config.constants import (
    FURNITURE_CLASSIFICATION,
    NUMBER_WORDS,
    SEATER_PATTERNS
)

# Configure logging
logger = logging.getLogger(__name__)

class StyleClassification:
    """ Extract classification information from furniture queries with fuzzy matching and spell correction """
    
    def __init__(self):
        """ Initialize the style classification extractor """
        
        # Classification mappings with synonyms and variations
        self.classification_mappings = FURNITURE_CLASSIFICATION
        
        # Build reverse mappings for quick lookup
        self.synonym_to_classification = {}
        self.all_synonyms = set()
        self._buildReverseMappings()
    
    def _buildReverseMappings(self):
        """ Build reverse mappings from synonyms to classifications """
        for category, classifications in self.classification_mappings.items():
            for classification, synonyms in classifications.items():
                # Map the classification name to itself
                key = f"{category}:{classification}"
                self.synonym_to_classification[classification.lower()] = key
                self.all_synonyms.add(classification.lower())
                
                # Map all synonyms to the classification
                for synonym in synonyms:
                    synonym_lower = synonym.lower()
                    self.synonym_to_classification[synonym_lower] = key
                    self.all_synonyms.add(synonym_lower)
    
    def _preprocessQuery(self, query: str) -> str:
        """
            Preprocesses the input query text to handle common linguistic and numerical variations, 
            standardizing the text for downstream parsing and matching.
            
            Args:
                query (str): The raw input query text.

            Returns:
                str: The preprocessed and standardized query string.
        """
        query = query.lower().strip()
        
        # Handle number words
        for word, digit in NUMBER_WORDS.items():
            query = re.sub(rf'\b{word}\b', digit, query)
        
        # Handle common patterns
        query = re.sub(r'\bfor\s+(\d+)\s+people?\b', r'\1 seater', query)
        query = re.sub(r'\b(\d+)\s+people?\b', r'\1 seater', query)
        query = re.sub(r'\bseats?\s+(\d+)\b', r'\1 seater', query)
        query = re.sub(r'\b(\d+)\s+person\b', r'\1 seater', query)
        
        # Handle sofa context
        query = re.sub(r'\bsofa\s+for\s+(\d+)\b', r'\1 seater sofa', query)
        
        return query
    
    def _extractFromText(self, text: str) -> Dict[str, List[str]]:
        """
            Extracts detailed classifications (sub-features, materials, sizes, etc.) from 
            preprocessed text by matching n-grams against known classification terms.

            Args:
                text (str): The preprocessed, lowercased input text query.

            Returns:
                Dict[str, List[str]]: A dictionary where keys are classification categories 
                            (e.g., 'material', 'size') and values are lists of unique, detected 
                            classification terms (e.g., ['leather', 'wood']). Empty categories are omitted.
        """
        detected_classifications = {}
        words = text.split()
        
        # Initialize result structure
        for category in self.classification_mappings:
            detected_classifications[category] = []
        
        # Try different n-gram window sizes
        for window_size in [1, 2, 3, 4]:
            for i in range(len(words) - window_size + 1):
                phrase = ' '.join(words[i:i + window_size])
                
                # Direct match
                if phrase in self.synonym_to_classification:
                    category_class = self.synonym_to_classification[phrase]
                    category, classification = category_class.split(':', 1)
                    if classification not in detected_classifications[category]:
                        detected_classifications[category].append(classification)
                        logger.debug(f"Direct match: '{phrase}' -> {classification}")
                
                # Fuzzy match for phrases longer than 3 characters
                elif len(phrase) > 5:
                    # Try fuzzy matching
                    fuzzy_match = ""
                    result = fuzzyMatch(phrase, self.all_synonyms, threshold=0.9)
                    if result:
                        fuzzy_match = result[0]
                    else:
                        pass
                    if fuzzy_match and fuzzy_match in self.synonym_to_classification:
                        category_class = self.synonym_to_classification[fuzzy_match]
                        category, classification = category_class.split(':', 1)
                        if classification not in detected_classifications[category]:
                            detected_classifications[category].append(classification)
                            logger.debug(f"Fuzzy match: '{phrase}' -> {fuzzy_match} -> {classification}")
        
        # Filter out empty categories
        return {k: v for k, v in detected_classifications.items() if v}
    
    def _extractContextualPatterns(self, text: str) -> Dict[str, List[str]]:
        """
            Extracts contextual classifications (like seating capacity) from the text 
            using predefined regular expression patterns and associated converter functions.

            Args:
                text (str): The preprocessed input text query.

            Returns:
                Dict[str, List[str]]: A dictionary containing the extracted classification, typically 
                            with the key 'VariantType' and a list of standardized variant strings (e.g., {'VariantType': ['3 Seater']}).
        """
        classifications = {}
        
        for pattern, converter in SEATER_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                classification = converter(match)
                if "VariantType" not in classifications:
                    classifications["VariantType"] = []
                if classification not in classifications["VariantType"]:
                    classifications["VariantType"].append(classification)
                    logger.debug(f"Pattern match: '{match.group()}' -> {classification}")
        
        return classifications
    
    def _validateClassifications(self, classifications: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
            Validates and cleans a dictionary of extracted classifications against a 
            predefined knowledge base.

            Args:
                classifications (Dict[str, List[str]]): A dictionary of classification 
                                    categories (keys) and lists of extracted terms (values).

            Returns:
                Dict[str, List[str]]: A cleaned dictionary containing only the validated 
                                    categories and classification terms. Empty categories are removed.
        """
        validated = {}
        
        for category, items in classifications.items():
            if category in self.classification_mappings:
                valid_items = []
                for item in items:
                    if item in self.classification_mappings[category]:
                        valid_items.append(item)
                if valid_items:
                    validated[category] = valid_items
        
        return validated
    
    def extractClassification(self, query: str) -> Dict[str, List[str]]:
        """
            Extract classification information from query
            
            Args:
                query (str): Input query to analyze
                
            Returns:
                Dict[str, List[str]]: Dictionary with classification categories and their values
        """
        # Step 1: Preprocess the query
        preprocessed = self._preprocessQuery(query)
        
        # Step 2: Apply spell correction
        spell_corrected = spellCorrect(preprocessed)
        
        # Step 3: Extract using direct/fuzzy matching
        text_classifications = self._extractFromText(spell_corrected)
        
        # Step 4: Extract using contextual patterns
        pattern_classifications = self._extractContextualPatterns(spell_corrected)

        final_classifications = {}
        all_categories = set(text_classifications.keys()) | set(pattern_classifications.keys())

        for category in all_categories:
            # Start with pattern matches (more reliable)
            final_classifications[category] = pattern_classifications.get(category, []).copy()

            # Add text matches only if not already covered
            for item in text_classifications.get(category, []):
                if item not in final_classifications[category]:
                    final_classifications[category].append(item)
        
        # Step 5: Validate and clean results
        validated_classifications = self._validateClassifications(final_classifications)
        
        return validated_classifications