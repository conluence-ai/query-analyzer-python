"""
StyleClassification - Standalone module for extracting variant classifications
Save this as: query_parser/style_classification.py
"""

import re
import logging
from typing import Dict, List, Optional
from difflib import SequenceMatcher

# Optional imports for better performance
try:
    import Levenshtein
except ImportError:
    Levenshtein = None
    
try:
    from spellchecker import SpellChecker
except ImportError:
    SpellChecker = None

# Configure logging
logger = logging.getLogger(__name__)

class StyleClassification:
    """Extract classification information from furniture queries with fuzzy matching and spell correction"""
    
    def __init__(self):
        """Initialize the style classification extractor"""
        
        # Classification mappings with synonyms and variations
        self.classification_mappings = {
            "VariantType": {
                # Seater variations
                "1 Seater": [
                    "1 seater", "single seater", "one seater", "1-seater", "single seat",
                    "solo", "individual", "personal", "single person", "1 person", "single"
                ],
                "2 Seater": [
                    "2 seater", "two seater", "double seater", "2-seater", "two seat",
                    "double seat", "couple", "loveseat", "love seat", "2 person",
                    "two person", "dual", "pair", "double sofa", "double"
                ],
                "3 Seater": [
                    "3 seater", "three seater", "triple seater", "3-seater", "three seat",
                    "triple seat", "3 person", "three person", "triple", "3 people",
                    "three people", "family", "triple sofa", "three"
                ],
                "4 Seater": [
                    "4 seater", "four seater", "quad seater", "4-seater", "four seat",
                    "quad seat", "4 person", "four person", "quad", "4 people",
                    "four people", "large family", "quad sofa", "four"
                ],
                
                # Size variations
                "Compact": [
                    "compact", "small", "mini", "tiny", "petite", "space saving",
                    "apartment size", "studio", "small space", "narrow", "slim"
                ],
                "Standard": [
                    "standard", "regular", "normal", "medium", "average", "typical",
                    "standard size", "regular size", "medium size"
                ],
                "Oversized": [
                    "oversized", "extra large", "xl", "xxl", "jumbo", "huge",
                    "massive", "big", "large size", "super size", "king size"
                ],
                "Small": [
                    "small", "little", "mini", "compact", "petite", "tiny",
                    "space efficient", "small size"
                ],
                "Medium": [
                    "medium", "mid", "middle", "moderate", "average size",
                    "medium size", "mid size", "standard"
                ],
                "Large": [
                    "large", "big", "spacious", "roomy", "generous", "wide",
                    "large size", "big size", "family size"
                ],
                "Extra Large": [
                    "extra large", "xl", "very large", "super large", "oversized",
                    "jumbo", "king size", "queen size", "massive"
                ]
            }
        }
        
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
        
        # Number word mappings
        self.number_words = {
            "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
            "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
            "single": "1", "double": "2", "triple": "3", "quad": "4"
        }
    
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
        for word, digit in self.number_words.items():
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
        
        # Seater patterns
        seater_patterns = [
            (r'\b(\d+)[\s-]?seater?\b', lambda m: f"{m.group(1)} Seater"),
            (r'\b(\d+)[\s-]?seat\b', lambda m: f"{m.group(1)} Seater"),
            (r'\b(\d+)[\s-]?person\b', lambda m: f"{m.group(1)} Seater"),
            (r'\b(\d+)[\s-]?people\b', lambda m: f"{m.group(1)} Seater"),
            (r'\bfor[\s-]?(\d+)\b', lambda m: f"{m.group(1)} Seater"),
        ]
        
        for pattern, converter in seater_patterns:
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
        logger.info(f"Extracting classification from: '{query}'")
        
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
        
        # Step 5: Merge results
        # final_classifications = {}
        # all_categories = set(text_classifications.keys()) | set(pattern_classifications.keys())
        
        # for category in all_categories:
        #     final_classifications[category] = []
            
        #     # Add from text extraction
        #     if category in text_classifications:
        #         final_classifications[category].extend(text_classifications[category])
            
        #     # Add from pattern extraction (avoid duplicates)
        #     if category in pattern_classifications:
        #         for item in pattern_classifications[category]:
        #             if item not in final_classifications[category]:
        #                 final_classifications[category].append(item)
        final_classifications = {}
        all_categories = set(text_classifications.keys()) | set(pattern_classifications.keys())

        for category in all_categories:
            # Start with pattern matches (more reliable)
            final_classifications[category] = pattern_classifications.get(category, []).copy()

            # Add text matches only if not already covered
            for item in text_classifications.get(category, []):
                if item not in final_classifications[category]:
                    final_classifications[category].append(item)
        
        # Step 6: Validate and clean results
        validated_classifications = self._validate_classifications(final_classifications)
        
        return validated_classifications