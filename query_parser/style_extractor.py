""" Style Extractor for furniture queries """

# Import necessary libraries
import re
from typing import List, Optional

# Import custom modules
from utils.helpers import spellCorrect, fuzzyMatch

# Import constants and mappings
from config.constants import (
    FURNITURE_STYLES,
    STYLE_CONTEXTUAL_PATTERNS
)

class StyleExtractor:
    """ Extract furniture styles from text using fuzzy matching and spell correction """
    
    def __init__(self):
        """ Initialize the style extractor """

        # Build mappings for efficient lookup
        self.synonym_to_style = {}
        self.all_style_terms = set()
        self._buildStyleMappings()
    
    def _buildStyleMappings(self):
        """ Build reverse mapping from synonyms to main styles """
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
        
    def _extractDirectMatches(self, text: str) -> List[str]:
        """
            Extracts design styles from the input text using direct keyword matching.

            Args:
                text (str): The raw text string (e.g., a search query) to scan for style keywords.

            Returns:
                List[str]: A list of unique, standardized style labels detected in the text.
        """
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
    
    def _extractFuzzyMatches(self, text: str) -> List[str]:
        """
            Extracts styles from the input text using fuzzy matching

            Args:
                text (str): The raw text string (e.g., a search query) to scan for 
                            potentially misspelled style keywords.

            Returns:
                List[str]: A list of unique, standardized style labels detected via fuzzy 
                        matching.
        """
        detected_styles = []
        corrected_text = spellCorrect(text)
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
                fuzzy_match = ""
                result = fuzzyMatch(phrase, self.all_style_terms, threshold=0.8)
                if result:
                    fuzzy_match = result[0]
                else:
                    pass
                if fuzzy_match and fuzzy_match in self.synonym_to_style:
                    style = self.synonym_to_style[fuzzy_match]
                    if style not in detected_styles:
                        detected_styles.append(style)
        
        return detected_styles
    
    def _extractContextualStylePatterns(self, text: str) -> List[str]:
        """
            Extracts design styles from the input text by matching against predefined 
            regular expression patterns and contextual phrases.

            Args:
                text (str): The raw text string (e.g., a search query) to scan for 
                            contextual style patterns.

            Returns:
                List[str]: A list of unique, standardized style labels derived from successful pattern matches.
        """
        detected_styles = []
        text_lower = text.lower()
        
        for pattern, style in STYLE_CONTEXTUAL_PATTERNS:
            if re.search(pattern, text_lower):
                if style not in detected_styles:
                    detected_styles.append(style)
        
        return detected_styles
    
    def _getStyleInfo(self, style_name: str) -> Optional[List[str]]:
        """
            Get synonyms for a specific style
            
            Args:
                style_name (str): Style name to look up
                
            Returns:
                Optional[List[str]]: List of synonyms or None if not found
        """
        return FURNITURE_STYLES.get(style_name)
    
    def _getAllStyles(self) -> List[str]:
        """
        Get all available style names
        
        Returns:
            List[str]: List of all main style names
        """
        return list(FURNITURE_STYLES.keys())
    
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
        direct_matches = self._extractDirectMatches(text)
        detected_styles.extend(direct_matches)
        
        # Method 2: Fuzzy matching with spell correction
        fuzzy_matches = self._extractFuzzyMatches(text)
        detected_styles.extend(fuzzy_matches)
        
        # Method 3: Contextual pattern matching
        contextual_matches = self._extractContextualStylePatterns(text)
        detected_styles.extend(contextual_matches)
        
        # Remove duplicates while preserving order
        unique_styles = []
        seen = set()
        for style in detected_styles:
            if style not in seen:
                unique_styles.append(style)
                seen.add(style)
        
        return unique_styles