"""Style Extractor for furniture queries with fuzzy matching and spell correction"""

# Import necessary libraries
import re
from typing import List, Set, Optional
from difflib import SequenceMatcher
import Levenshtein  # pip install python-Levenshtein
from spellchecker import SpellChecker  # pip install pyspellchecker

# Import constants and mappings
from config.constants import (
    FURNITURE_STYLES,
    STYLE_CONTEXTUAL_PATTERNS
)

class StyleExtractor:
    """Extract furniture styles from text using fuzzy matching and spell correction"""
    
    def __init__(self):
        """Initialize the style extractor"""
        # Fuzzy matching threshold
        self.fuzzy_threshold = 0.8
        
        # Spell checker
        self.spell = SpellChecker(distance=2)
        
        # Build mappings for efficient lookup
        self.synonym_to_style = {}
        self.all_style_terms = set()
        self._build_style_mappings()
    
    def _build_style_mappings(self):
        """Build reverse mapping from synonyms to main styles"""
        for main_style, synonyms in FURNITURE_STYLES.items():
            # Map the main style name to itself (case-insensitive)
            main_style_lower = main_style.lower()
            self.synonym_to_style[main_style_lower] = main_style
            self.all_style_terms.add(main_style_lower)
            
            # Map all synonyms to the main style
            for synonym in synonyms:
                synonym_lower = synonym.lower()
                self.synonym_to_style[synonym_lower] = main_style
                self.all_style_terms.add(synonym_lower)
    
    def _spell_correct(self, text: str) -> str:
        """Apply spell correction to text"""
        corrected = []
        for word in text.split():
            if len(word) > 2:  # Don't correct very short words
                corrected_word = self.spell.correction(word)
                corrected.append(corrected_word if corrected_word else word)
            else:
                corrected.append(word)
        return " ".join(corrected)
    
    def _fuzzy_match(self, phrase: str, candidates: Set[str]) -> Optional[str]:
        """Find the best fuzzy match for a phrase"""
        best_match = None
        best_score = 0
        
        for candidate in candidates:
            try:
                # Use Levenshtein distance if available, otherwise SequenceMatcher
                similarity = Levenshtein.ratio(phrase.lower(), candidate.lower())
            except (ImportError, NameError):
                similarity = SequenceMatcher(None, phrase.lower(), candidate.lower()).ratio()
            
            if similarity > best_score and similarity >= self.fuzzy_threshold:
                best_score = similarity
                best_match = candidate
        
        return best_match
    
    def _extract_direct_matches(self, text: str) -> List[str]:
        """Extract styles using direct keyword matching"""
        detected_styles = []
        text_lower = text.lower()
        
        # Check for multi-word phrases first (longer matches are more specific)
        for window_size in [3, 2, 1]:
            words = text_lower.split()
            for i in range(len(words) - window_size + 1):
                phrase = ' '.join(words[i:i + window_size])
                
                if phrase in self.synonym_to_style:
                    style = self.synonym_to_style[phrase]
                    if style not in detected_styles:
                        detected_styles.append(style)
        
        return detected_styles
    
    def _extract_fuzzy_matches(self, text: str) -> List[str]:
        """Extract styles using fuzzy matching for misspellings"""
        detected_styles = []
        corrected_text = self._spell_correct(text)
        words = corrected_text.lower().split()
        
        # Try different window sizes for phrase matching
        for window_size in [3, 2, 1]:
            for i in range(len(words) - window_size + 1):
                phrase = ' '.join(words[i:i + window_size])
                
                # Skip if already found exact match
                if phrase in self.synonym_to_style:
                    continue
                
                # Skip very short phrases for fuzzy matching
                if len(phrase) < 3:
                    continue
                
                # Try fuzzy matching
                fuzzy_match = self._fuzzy_match(phrase, self.all_style_terms)
                if fuzzy_match and fuzzy_match in self.synonym_to_style:
                    style = self.synonym_to_style[fuzzy_match]
                    if style not in detected_styles:
                        detected_styles.append(style)
        
        return detected_styles
    
    def _extract_contextual_patterns(self, text: str) -> List[str]:
        """Extract styles using contextual patterns and common phrases"""
        detected_styles = []
        text_lower = text.lower()
        
        for pattern, style in STYLE_CONTEXTUAL_PATTERNS:
            if re.search(pattern, text_lower):
                if style not in detected_styles:
                    detected_styles.append(style)
        
        return detected_styles
    
    def extractStyles(self, text: str) -> List[str]:
        """
        Main method to extract styles from text
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            List[str]: List of detected style names
        """
        detected_styles = []
        
        # Method 1: Direct exact matching
        direct_matches = self._extract_direct_matches(text)
        detected_styles.extend(direct_matches)
        
        # Method 2: Fuzzy matching with spell correction
        fuzzy_matches = self._extract_fuzzy_matches(text)
        detected_styles.extend(fuzzy_matches)
        
        # Method 3: Contextual pattern matching
        contextual_matches = self._extract_contextual_patterns(text)
        detected_styles.extend(contextual_matches)
        
        # Remove duplicates while preserving order
        unique_styles = []
        seen = set()
        for style in detected_styles:
            if style not in seen:
                unique_styles.append(style)
                seen.add(style)
        
        return unique_styles
    
    def get_style_info(self, style_name: str) -> Optional[List[str]]:
        """
        Get synonyms for a specific style
        
        Args:
            style_name (str): Style name to look up
            
        Returns:
            Optional[List[str]]: List of synonyms or None if not found
        """
        return FURNITURE_STYLES.get(style_name)
    
    def get_all_styles(self) -> List[str]:
        """
        Get all available style names
        
        Returns:
            List[str]: List of all main style names
        """
        return list(FURNITURE_STYLES.keys())