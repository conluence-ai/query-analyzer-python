""" Feature Extractor for furniture queries """

# Import necessary libraries
import re
import logging
from typing import List

# Import custom modules
from utils.helpers import spellCorrect, fuzzyMatch

# Import constants and mappings
from config.constants import (
    FURNITURE_CATEGORY,
    CATEGORY_MAPPINGS,
    FEATURE_CONTEXTUAL_PATTERNS,
    FUZZY_PATTERNS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """ Extract furniture features from text using fuzzy matching and spell correction """

    def __init__(self):
        """
            Initialize the feature extractor
        """
        # Fuzzy matching threshold
        self.all_synonyms = set()  # For fuzzy matching

        # Feature dictionary and mappings
        self.synonym_to_feature = {}
        self.feature_to_category = {}
        
        # Build mappings from your FURNITURE_CATEGORY
        self._buildFeatureMappings()

    def _buildFeatureMappings(self):
        """ Build reverse mapping from synonyms to main features """
        for main_feature, synonyms in FURNITURE_CATEGORY.items():
            # Map the main feature to itself
            self.synonym_to_feature[main_feature.lower()] = main_feature
            self.all_synonyms.add(main_feature.lower())
            
            # Map all synonyms to the main feature
            for synonym in synonyms:
                synonym_lower = synonym.lower()
                self.synonym_to_feature[synonym_lower] = main_feature
                self.all_synonyms.add(synonym_lower)
            
            # Assign category
            category = CATEGORY_MAPPINGS.get(main_feature, 'Other')
            self.feature_to_category[main_feature] = category

    def _getCategoriesFromText(self, text: str) -> List[str]:
        """
            Extract features by direct keyword matching and fuzzy matching
            
            Args:
                text (str): The raw input text (e.g., a search query) to scan for features.

            Returns:
                List[str]: A list of unique, standardized feature labels detected through 
                        contextual pattern matching.    
        """
        detected = []
        words = text.split()

        # Step 0: Spell correction
        corrected_text = spellCorrect(text)
        corrected_words = corrected_text.split()
        
        # First pass: Exact match
        for window_size in [1, 2, 3]:
            for i in range(len(corrected_words) - window_size + 1):
                phrase = ' '.join(corrected_words[i:i + window_size])

                # normalize common suffixes
                phrase = re.sub(r'(shaped|shapes)$', 'shape', phrase)

                if phrase in self.synonym_to_feature:
                    main_feature = self.synonym_to_feature[phrase]
                    if self._hasContextualMatch(phrase, corrected_words, i):
                        if main_feature not in detected:
                            detected.append(main_feature)

        # Second pass: Fuzzy matching for n-grams
        for window_size in [2, 3, 1]:  # prefer multi-word matches first
            for i in range(len(corrected_words) - window_size + 1):
                phrase = ' '.join(corrected_words[i:i + window_size])

                if len(phrase) < 3:  # skip too-short phrases
                    continue

                if phrase not in self.synonym_to_feature:
                    # Try fuzzy matching
                    fuzzy_match = ""
                    result = fuzzyMatch(phrase, self.all_synonyms, threshold=0.95)
                    if result:
                        fuzzy_match = result[0]
                    else:
                        pass
                    if fuzzy_match:
                        main_feature = self.synonym_to_feature[fuzzy_match]
                        if self._hasContextualMatch(phrase, corrected_words, i):
                            if main_feature not in detected:
                                detected.append(main_feature)
        
        return detected
    
    def _hasContextualMatch(self, phrase: str, words: List[str], position: int) -> bool:
        """
            Checks if a detected feature phrase is contextually relevant based on its 
            surrounding words in the sentence.

            Args:
                phrase (str): The specific feature phrase that was detected (e.g., "genuine leather").
                words (List[str]): The tokenized list of all words in the input text.
                position (int): The starting index of the phrase within the `words` list.

            Returns:
                bool: True if the phrase is deemed contextually relevant (or if no specific 
                    contextual check is defined for it), False otherwise.
        """
        if "leather" in phrase:
            context_window = words[max(0, position - 2):min(len(words), position + 3)]
            furniture_parts = ['sofa', 'chair', 'back', 'seat', 'arm', 'cushion']
            return any(part in ' '.join(context_window).lower() for part in furniture_parts)
        return True
    
    def _extractContextualCategories(self, text: str) -> List[str]:
        """
            Extracts features by matching the input text against predefined contextual 
            phrase patterns, incorporating support for common misspellings using fuzzy patterns.

            Args:
                text (str): The raw input text (e.g., a search query) to scan for features.

            Returns:
                List[str]: A list of unique, standardized feature labels detected through 
                        contextual pattern matching.
        """
        detected = []

        # Apply regular patterns
        for pattern, feature in FEATURE_CONTEXTUAL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                # Normalize feature to lowercase key in FURNITURE_CATEGORY
                feature_lower = feature.lower()
                if feature_lower in FURNITURE_CATEGORY and feature_lower not in detected:
                    detected.append(feature_lower)

        # Apply fuzzy patterns
        for pattern, corrected_word in FUZZY_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                corrected_text = text.replace(match.group(), corrected_word)
                for context_pattern, feature in FEATURE_CONTEXTUAL_PATTERNS:
                    if re.search(context_pattern, corrected_text, re.IGNORECASE):
                        feature_lower = feature.lower()
                        if feature_lower in FURNITURE_CATEGORY and feature_lower not in detected:
                            detected.append(feature_lower)

        return detected
    
    def extractFeatures(self, text: str) -> List[str]:
        """
            Improved feature extraction returning a list directly
            
            Args:
                text: Input text to extract features from
                
            Returns:
                List of detected features (no duplicates)
        """
        detected_features = []
        text_lower = text.lower()
        
        # Method 1: Direct keyword matching with fuzzy support
        keyword_features = self._getCategoriesFromText(text_lower)
        detected_features.extend(keyword_features)
        
        # # Method 2: Contextual phrase matching
        # contextual_features = self._extractContextualCategories(text_lower)
        # detected_features.extend(contextual_features)
        # --- Disambiguation: prevent both "l shape" and "c shape" from appearing together ---
        if any("l shape" in f for f in detected_features) and any("c shape" in f for f in detected_features):
            text_lower = text.lower()
            if "l" in text_lower and not "c" in text_lower:
                detected_features = [f for f in detected_features if f != "c shape"]
            elif "c" in text_lower and not "l" in text_lower:
                detected_features = [f for f in detected_features if f != "l shape"]
        
        # Remove duplicates while preserving order
        unique_features = []
        seen = set()
        for feature in detected_features:
            if feature not in seen:
                unique_features.append(feature)
                seen.add(feature)
        
        return unique_features