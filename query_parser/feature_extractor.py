""" Feature Extractor for furniture queries """

# Import necessary libraries
import re
import logging
from typing import List
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

    def _normalizeTense(self, word: str) -> List[str]:
        """
        Generate common word forms for better matching.
        Returns list of variations to check.
        
        Args:
            word: Single word to normalize
            
        Returns:
            List of word variations (original + common forms)
        """
        variations = [word]
        
        # Handle -ed suffix (rounded -> round)
        if word.endswith('ed'):
            base = word[:-2]
            variations.append(base)
            # Handle doubled consonants (padded -> pad)
            if len(base) >= 2 and base[-1] == base[-2]:
                variations.append(base[:-1])
        
        # Handle -ing suffix (reclining -> recline)
        if word.endswith('ing'):
            base = word[:-3]
            variations.append(base)
            variations.append(base + 'e')  # recline
            
        # Add -ed form if not already present (round -> rounded)
        if not word.endswith('ed'):
            variations.append(word + 'ed')
            # Handle doubled consonants (pad -> padded)
            if len(word) >= 2 and word[-1] in 'bdfglmnprst' and word[-2] not in 'aeiou':
                variations.append(word + word[-1] + 'ed')
        
        # Add -ing form
        if not word.endswith('ing'):
            variations.append(word + 'ing')
            if word.endswith('e'):
                variations.append(word[:-1] + 'ing')
        
        return variations

    def _normalizePhrase(self, phrase: str) -> List[str]:
        """
        Generate variations of a phrase by normalizing each word.
        
        Args:
            phrase: Multi-word phrase
            
        Returns:
            List of phrase variations
        """
        words = phrase.split()
        if len(words) == 1:
            return self._normalizeTense(words[0])
        
        # For multi-word phrases, normalize each word and combine
        all_variations = [[word] + self._normalizeTense(word) for word in words]
        
        # Generate combinations (limit to avoid explosion)
        phrase_variations = [phrase]  # original
        
        # Try normalizing each position
        for i, word_vars in enumerate(all_variations):
            for var in word_vars[1:]:  # skip original
                new_phrase = words.copy()
                new_phrase[i] = var
                phrase_variations.append(' '.join(new_phrase))
        
        return phrase_variations

    def _contextAwareSpellCorrect(self, text: str) -> str:
        """
        Apply spell correction with context awareness for furniture terms.
        
        Args:
            text: Input text
            
        Returns:
            Corrected text
        """
        words = text.lower().split()
        corrected_words = []
        
        for i, word in enumerate(words):
            # Get context (previous and next word)
            prev_word = words[i-1] if i > 0 else ""
            next_word = words[i+1] if i < len(words) - 1 else ""
            
            # Apply spell correction
            corrected = spellCorrect(word)
            
            # Context-based correction override
            # If the corrected word doesn't make sense in furniture context,
            # try fuzzy matching against known terms
            if corrected != word:
                # Check if corrected word + context matches any furniture feature
                context_phrase = f"{prev_word} {corrected} {next_word}".strip()
                
                # Try fuzzy match on the context phrase
                for window_size in [3, 2, 1]:
                    phrase_parts = context_phrase.split()
                    if len(phrase_parts) >= window_size:
                        for j in range(len(phrase_parts) - window_size + 1):
                            test_phrase = ' '.join(phrase_parts[j:j + window_size])
                            
                            # Check if any known synonym is close
                            match_result = fuzzyMatch(test_phrase, self.all_synonyms, threshold=0.85)
                            if match_result:
                                # Found a better match in furniture context
                                matched_term = match_result[0]
                                matched_words = matched_term.split()
                                
                                # If our corrected word is part of this match, use it
                                if len(matched_words) > 1 and corrected in matched_words:
                                    # Use the corresponding word from the matched term
                                    idx = matched_words.index(corrected) if corrected in matched_words else -1
                                    if idx >= 0:
                                        corrected = matched_words[idx]
                                    break
            
            corrected_words.append(corrected)
        
        return ' '.join(corrected_words)

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

        # Step 0: Context-aware spell correction
        corrected_text = self._contextAwareSpellCorrect(text)
        corrected_words = corrected_text.split()
        
        # First pass: Exact match with tense normalization
        for window_size in [1, 2, 3]:
            for i in range(len(corrected_words) - window_size + 1):
                phrase = ' '.join(corrected_words[i:i + window_size])

                # normalize common suffixes
                phrase = re.sub(r'(shaped|shapes)$', 'shape', phrase)

                # Try original phrase first
                if phrase in self.synonym_to_feature:
                    main_feature = self.synonym_to_feature[phrase]
                    if self._hasContextualMatch(phrase, corrected_words, i):
                        if main_feature not in detected:
                            detected.append(main_feature)
                else:
                    # Try normalized versions
                    phrase_variations = self._normalizePhrase(phrase)
                    for variation in phrase_variations:
                        if variation in self.synonym_to_feature:
                            main_feature = self.synonym_to_feature[variation]
                            if self._hasContextualMatch(phrase, corrected_words, i):
                                if main_feature not in detected:
                                    detected.append(main_feature)
                                    break

        # Second pass: Fuzzy matching for n-grams
        for window_size in [2, 3, 1]:  # prefer multi-word matches first
            for i in range(len(corrected_words) - window_size + 1):
                phrase = ' '.join(corrected_words[i:i + window_size])

                if len(phrase) < 3:  # skip too-short phrases
                    continue

                if phrase not in self.synonym_to_feature:
                    # Try fuzzy matching on normalized versions
                    best_match = None
                    best_score = 0
                    
                    phrase_variations = self._normalizePhrase(phrase)
                    for variation in phrase_variations:
                        result = fuzzyMatch(variation, self.all_synonyms, threshold=0.85)
                        if result:
                            # fuzzyMatch returns (match, score)
                            match_str = result[0]
                            score = result[1] if len(result) > 1 else 1.0
                            if score > best_score:
                                best_match = match_str
                                best_score = score
                    
                    if best_match:
                        main_feature = self.synonym_to_feature[best_match]
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

    def _applyCorrection(self, original_text: str, text_clean: str, matched_categories: List[str], locked_product_terms: List[str] = ['armchair']) -> str:
        """
        Corrects spelling mistakes in feature-related words (not product types)
        by fuzzy-matching against known feature synonyms and detected feature terms.
        """
        corrected_text = original_text
        locked_product_terms = [w.lower() for w in (locked_product_terms or [])]

        # 🚫 Exclude product-type terms from correction
        try:
            from config.constants import FURNITURE_TYPE
            excluded_terms = set()
            for main_cat, synonyms in FURNITURE_TYPE.items():
                excluded_terms.add(main_cat.lower())
                for syn in synonyms:
                    excluded_terms.add(syn.lower())
        except ImportError:
            excluded_terms = set()

        # ✅ Build correction targets (features only)
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

            # Skip locked product words (e.g., "armchair", "sofa")
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
        corrected_query = text
        text_clean = text.lower().strip()
        
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
            if feature not in seen and '_' not in feature:
                unique_features.append(feature)
                seen.add(feature)
        corrected_query = self._applyCorrection(text, text_clean, unique_features)
        return unique_features, corrected_query