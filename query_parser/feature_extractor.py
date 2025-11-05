""" Feature Extractor for furniture queries """

# Import necessary libraries
import re
import logging
from typing import List, Tuple
from difflib import SequenceMatcher

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
        """ Build reverse mapping from main features only (synonyms disabled) """
        for main_feature, synonyms in FURNITURE_CATEGORY.items():
            # Map ONLY the main feature to itself (synonyms ignored)
            self.synonym_to_feature[main_feature.lower()] = main_feature
            self.all_synonyms.add(main_feature.lower())
            
            # Synonyms are completely ignored
            # To re-enable, uncomment below:
            # for synonym in synonyms:
            #     synonym_lower = synonym.lower()
            #     self.synonym_to_feature[synonym_lower] = main_feature
            #     self.all_synonyms.add(synonym_lower)
            
            # Assign category
            category = CATEGORY_MAPPINGS.get(main_feature, 'Other')
            self.feature_to_category[main_feature] = category

    def _getCategoriesFromText(self, text: str) -> List[str]:
        """
            Extract features using greedy longest-match-first algorithm
            to prevent overlapping and redundant matches
            
            Args:
                text (str): The preprocessed input text

            Returns:
                List[str]: List of unique detected features
        """
        words = text.split()
        word_count = len(words)
        
        logger.debug(f"\n{'='*60}")
        logger.debug(f"EXTRACTING FEATURES FROM: '{text}'")
        logger.debug(f"WORDS: {words} (indices 0-{word_count-1})")
        logger.debug(f"{'='*60}")
        
        # Track which word indices have been consumed
        consumed_indices = set()
        detected_features = []
        
        # PHASE 1: Greedy matching (longest phrases first)
        for window_size in [4, 3, 2, 1]:
            logger.debug(f"\n--- Checking window size {window_size} ---")
            
            for start_idx in range(word_count - window_size + 1):
                # Skip if any word in this window is already consumed
                window_indices = list(range(start_idx, start_idx + window_size))
                if any(idx in consumed_indices for idx in window_indices):
                    phrase = ' '.join(words[start_idx:start_idx + window_size])
                    logger.debug(f"  SKIP: '{phrase}' (indices {window_indices}) - overlaps consumed: {consumed_indices}")
                    continue
                
                phrase = ' '.join(words[start_idx:start_idx + window_size])
                matched_feature = None
                match_type = None
                
                logger.debug(f"  CHECK: '{phrase}' (indices {window_indices})")
                
                # Try direct match
                if phrase in self.synonym_to_feature:
                    matched_feature = self.synonym_to_feature[phrase]
                    match_type = "direct"
                    logger.debug(f"    → Direct match found!")
                
                # Try fuzzy match for longer phrases
                elif len(phrase) > 6:
                    fuzzy_threshold = 0.96 if ("detail" in phrase or "metal" in phrase) else 0.93
                    result = fuzzyMatch(phrase, self.all_synonyms, threshold=fuzzy_threshold)
                    
                    if result:
                        fuzzy_match = result[0]
                        if fuzzy_match in self.synonym_to_feature:
                            matched_feature = self.synonym_to_feature[fuzzy_match]
                            match_type = "fuzzy"
                            logger.debug(f"    → Fuzzy match found: '{fuzzy_match}'")
                
                # If we found a match, record it
                if matched_feature:
                    # Check contextual relevance
                    if self._hasContextualMatch(phrase, words, start_idx):
                        if matched_feature not in detected_features:
                            detected_features.append(matched_feature)
                            
                            # Mark these indices as consumed
                            for idx in window_indices:
                                consumed_indices.add(idx)
                            
                            logger.debug(f"  ✓✓✓ {match_type.upper()} MATCH: '{phrase}' -> {matched_feature}")
                            logger.debug(f"       Consumed indices: {window_indices}")
                            logger.debug(f"       Total consumed: {sorted(consumed_indices)}")
                    else:
                        logger.debug(f"    → Match rejected by contextual check")
                else:
                    logger.debug(f"    → No match found")
        
        logger.debug(f"\n{'='*60}")
        logger.debug(f"FINAL FEATURES: {detected_features}")
        logger.debug(f"{'='*60}\n")
        
        return detected_features
    
    def _hasContextualMatch(self, phrase: str, words: List[str], position: int) -> bool:
        """
            Checks if a detected feature phrase is contextually relevant

            Args:
                phrase (str): The detected feature phrase
                words (List[str]): All words in the input text
                position (int): Starting index of the phrase

            Returns:
                bool: True if contextually relevant
        """
        if "leather" in phrase:
            context_window = words[max(0, position - 2):min(len(words), position + 3)]
            furniture_parts = ['sofa', 'chair', 'back', 'seat', 'arm', 'cushion']
            return any(part in ' '.join(context_window).lower() for part in furniture_parts)
        return True
    
    def _extractContextualCategories(self, text: str) -> List[str]:
        """
            Extracts features by matching against contextual patterns

            Args:
                text (str): The raw input text

            Returns:
                List[str]: List of detected features from patterns
        """
        detected = []

        # Apply regular patterns
        for pattern, feature in FEATURE_CONTEXTUAL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
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

    def _applyCorrection(self, original_text: str, text_clean: str, matched_categories: List[str], locked_product_terms: List[str] = ['armchair']) -> str:
        """
        Corrects spelling mistakes in feature-related words
        """
        corrected_text = original_text
        locked_product_terms = [w.lower() for w in (locked_product_terms or [])]

        # Exclude product-type terms from correction
        try:
            from config.constants import FURNITURE_TYPE
            excluded_terms = set()
            for main_cat, synonyms in FURNITURE_TYPE.items():
                excluded_terms.add(main_cat.lower())
                for syn in synonyms:
                    excluded_terms.add(syn.lower())
        except ImportError:
            excluded_terms = set()

        # Build correction targets (features only)
        correction_targets = set()
        for synonym in self.all_synonyms:
            for part in synonym.split():
                if part.lower() not in excluded_terms:
                    correction_targets.add(part.lower())

        for cat in matched_categories:
            for part in cat.split():
                if part.lower() not in excluded_terms:
                    correction_targets.add(part.lower())

        shape_terms = ["circular", "round", "rectangular", "square", "oval", "inclined"]
        correction_targets.update(shape_terms)

        words = re.findall(r'\b\w{3,}\b', text_clean)

        for word in words:
            word_lower = word.lower()

            # Skip locked product words
            if any(word_lower in lp for lp in locked_product_terms):
                continue

            best_match, best_score = None, 0.0
            for target in correction_targets:
                score = SequenceMatcher(None, word_lower, target).ratio()
                if score > best_score:
                    best_score = score
                    best_match = target

            if best_match and best_score >= 0.7 and word_lower != best_match:
                if best_match not in excluded_terms:
                    corrected_text = re.sub(
                        r'\b' + re.escape(word) + r'\b',
                        best_match,
                        corrected_text,
                        flags=re.IGNORECASE
                    )

        return corrected_text
    
    def extractFeatures(self, text: str) -> Tuple[List[str], str]:
        """
            Improved feature extraction with greedy matching
            
            Args:
                text: Input text to extract features from
                
            Returns:
                Tuple of (detected features list, corrected query)
        """
        text_lower = text.lower()
        text_clean = text.lower().strip()
        
        # Extract features using greedy algorithm
        detected_features = self._getCategoriesFromText(text_lower)
        
        # Disambiguation: prevent both "l shape" and "c shape" from appearing together
        if any("l shape" in f for f in detected_features) and any("c shape" in f for f in detected_features):
            if "l" in text_lower and "c" not in text_lower:
                detected_features = [f for f in detected_features if f != "c shape"]
            elif "c" in text_lower and "l" not in text_lower:
                detected_features = [f for f in detected_features if f != "l shape"]
        
        # Apply spelling correction
        corrected_query = self._applyCorrection(text, text_clean, detected_features)
        
        return detected_features, corrected_query