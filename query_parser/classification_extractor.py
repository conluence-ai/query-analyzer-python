# Import necessary libraries
import re
import logging
from typing import Dict, List, Optional
from difflib import SequenceMatcher
import Levenshtein
from spellchecker import SpellChecker

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
        """Initialize the style classification extractor"""
        
        # Classification mappings with synonyms and variations
        self.classification_mappings = FURNITURE_CLASSIFICATION
        
        # Fuzzy matching threshold
        self.fuzzy_threshold = 0.9
        
        # Initialize spell checker if available
        if SpellChecker:
            self.spell = SpellChecker(distance=2)
        else:
            self.spell = None
        
        # Build reverse mappings for quick lookup
        self.synonym_to_classification = {}
        self.all_synonyms = set()
        self._build_reverse_mappings()
    
    def _build_reverse_mappings(self):
        """Build reverse mappings from synonyms to classifications"""
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
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess query to handle common variations"""
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
    
    def _spell_correct(self, text: str) -> str:
        """Apply spell correction to the text"""
        if not self.spell:
            return text
            
        corrected_words = []
        for word in text.split():
            if len(word) > 2:  # Only correct words longer than 2 chars
                corrected = self.spell.correction(word)
                corrected_words.append(corrected if corrected else word)
            else:
                corrected_words.append(word)
        return " ".join(corrected_words)
    
    def _fuzzy_match(self, phrase: str, candidates: set) -> Optional[str]:
        """Perform fuzzy matching against known synonyms"""
        best_match = None
        best_score = 0
        
        for candidate in candidates:
            try:
                if Levenshtein:
                    similarity = Levenshtein.ratio(phrase.lower(), candidate.lower())
                else:
                    similarity = SequenceMatcher(None, phrase.lower(), candidate.lower()).ratio()
            except:
                similarity = SequenceMatcher(None, phrase.lower(), candidate.lower()).ratio()
            
            if similarity > best_score and similarity >= self.fuzzy_threshold:
                best_score = similarity
                best_match = candidate
        
        return best_match
    
    def _extract_from_text(self, text: str) -> Dict[str, List[str]]:
        """Extract classifications from preprocessed text"""
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
                    fuzzy_match = self._fuzzy_match(phrase, self.all_synonyms)
                    if fuzzy_match and fuzzy_match in self.synonym_to_classification:
                        category_class = self.synonym_to_classification[fuzzy_match]
                        category, classification = category_class.split(':', 1)
                        if classification not in detected_classifications[category]:
                            detected_classifications[category].append(classification)
                            logger.debug(f"Fuzzy match: '{phrase}' -> {fuzzy_match} -> {classification}")
        
        # Filter out empty categories
        return {k: v for k, v in detected_classifications.items() if v}
    
    def _extract_contextual_patterns(self, text: str) -> Dict[str, List[str]]:
        """Extract using regex patterns for common contexts"""
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
    
    def _validate_classifications(self, classifications: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Validate and clean the extracted classifications"""
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
        preprocessed = self._preprocess_query(query)
        logger.debug(f"Preprocessed query: '{preprocessed}'")
        
        # Step 2: Apply spell correction
        spell_corrected = self._spell_correct(preprocessed)
        logger.debug(f"Spell corrected: '{spell_corrected}'")
        
        # Step 3: Extract using direct/fuzzy matching
        text_classifications = self._extract_from_text(spell_corrected)
        
        # Step 4: Extract using contextual patterns
        pattern_classifications = self._extract_contextual_patterns(spell_corrected)

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
        validated_classifications = self._validate_classifications(final_classifications)
        
        return validated_classifications