""" Brand and Product Extractor for furniture queries """

# Import necessary libraries
import re
import logging
from typing import List, Dict, Optional, Tuple
from difflib import SequenceMatcher
import Levenshtein
from spellchecker import SpellChecker

# Import custom module
from config.database import DatabaseManager
from utils.helpers import fuzzyMatch

# Import constants and mappings
from config.constants import BRAND_TABLE, PRODUCT_TABLE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

class BrandProductExtractor:
    """ Extract brand and product names from furniture queries using fuzzy matching and spell correction """
    
    def __init__(self):
        """ Initialize the brand and product extractor """
        # Fuzzy matching thresholds
        self.fuzzy_threshold = 0.85
        self.loose_threshold = 0.70  # Lowered for better spell-corrected matching
        
        # Minimum word length for partial matching (prevents "Mid" matching "MIDJ")
        self.min_word_length_for_partial = 5
        
        # Spell checker
        self.spell = SpellChecker(distance=2)
        
        # Database manager
        self.db = DatabaseManager(use_cache=True, use_connection_pool=True)
        
        # Load product and brand data from database
        self.product_names = {}  # lowercase -> original name
        self.brand_names = {}    # lowercase -> original name
        self._loadDataFromDB()
        
        # Common prepositions and keywords that indicate brand/product relationships
        self.brand_indicators = ['by', 'from', 'of', 'made by', 'designed by', 'brand']
        self.product_indicators = ['product', 'item', 'piece', 'furniture', 'want', 'need', 'looking for']
        
        # Common descriptive words that should NOT be matched as brands
        self.common_descriptors = {
            'mid', 'century', 'modern', 'classic', 'traditional', 'contemporary',
            'vintage', 'retro', 'industrial', 'rustic', 'minimalist', 'luxury',
            'premium', 'elegant', 'comfortable', 'soft', 'hard', 'inclined',
            'straight', 'curved', 'round', 'square', 'platform', 'modular',
            'sectional', 'reclining', 'adjustable', 'fixed', 'mobile', 'static',
            'large', 'small', 'medium', 'tall', 'short', 'wide', 'narrow',
            'deep', 'shallow', 'high', 'low', 'corner', 'center', 'side'
        }
    
    def _loadDataFromDB(self):
        """ Load product and brand names from database (with caching) """
        try:
            # Fetch product names (will use cache if available)
            products = self.db.fetchProductNames(use_cache=True)
            self.product_names = {p.lower(): p for p in products if p}
            
            # Fetch brand names (will use cache if available)
            brands = self.db.fetchBrandNames(use_cache=True)
            self.brand_names = {b.lower(): b for b in brands if b}
            
            # Add to spell checker
            self.spell.word_frequency.load_words(list(self.brand_names.keys()))
            self.spell.word_frequency.load_words(list(self.product_names.keys()))
            
            for name in list(self.brand_names.keys()) + list(self.product_names.keys()):
                for word in name.split():
                    if len(word) > 2:
                        self.spell.word_frequency.load_words([word])
            
            logger.info(f"Loaded {len(self.product_names)} products and {len(self.brand_names)} brands")
        except Exception as e:
            logger.error(f"Error loading products and brands: {e}")
            self.product_names = {}
            self.brand_names = {}
    
    def _refreshData(self):
        """
            Refresh product and brand data from database.
            This will invalidate cache and fetch fresh data.
        """
        # Clear cache for these tables
        self.db.clearCache(BRAND_TABLE)
        self.db.clearCache(PRODUCT_TABLE)
        
        # Reload data
        self._loadDataFromDB()
    
    def _calculateSimilarity(self, str1: str, str2: str) -> float:
        """
            Calculate similarity score between two strings
            
            Args:
                str1 (str): First string
                str2 (str): Second string
                
            Returns:
                float: Similarity score between 0 and 1
        """
        try:
            # Use Levenshtein distance if available (more accurate)
            return Levenshtein.ratio(str1.lower(), str2.lower())
        except (ImportError, NameError, AttributeError):
            # Fallback to SequenceMatcher
            return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
    
    def _isCommonDescriptor(self, word: str) -> bool:
        """
            Check if a word is a common descriptor that shouldn't be matched as a brand
            
            Args:
                word (str): Word to check
                
            Returns:
                bool: True if it's a common descriptor
        """
        word_lower = word.lower()
        
        # Check exact match
        if word_lower in self.common_descriptors:
            return True
        
        # Check if word is part of common multi-word descriptors
        multi_word_descriptors = ['mid century', 'inclined legs', 'with platform']
        for descriptor in multi_word_descriptors:
            if word_lower in descriptor:
                return True
        
        return False
    
    def _spellCorrectText(self, text: str) -> str:
        """
            Apply spell correction to text while preserving brand/product names
            
            Args:
                text (str): Input text to correct
                
            Returns:
                str: Spell-corrected text
        """
        corrected_words = []
        words = text.split()
        
        for word in words:
            # Skip very short words and punctuation
            if len(word) <= 2:
                corrected_words.append(word)
                continue
            
            # Clean word of punctuation for checking
            clean_word = re.sub(r'[^\w\s]', '', word).lower()
            
            # Skip common descriptors from spell correction
            if self._isCommonDescriptor(clean_word):
                corrected_words.append(word)
                continue
            
            # Check if word is a known brand or product (or part of one)
            is_known = False
            for brand_lower in self.brand_names.keys():
                if clean_word in brand_lower.split():
                    is_known = True
                    break
            
            if not is_known:
                for product_lower in self.product_names.keys():
                    if clean_word in product_lower.split():
                        is_known = True
                        break
            
            # If it's a known word, keep it as is
            if is_known:
                corrected_words.append(word)
            else:
                # Apply spell correction
                corrected = self.spell.correction(clean_word)
                if corrected and corrected != clean_word:
                    # Preserve original capitalization pattern
                    if word[0].isupper():
                        corrected = corrected.capitalize()
                    corrected_words.append(corrected)
                else:
                    corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    def _extractBrandWithIndicators(self, text: str) -> Optional[Tuple[str, float]]:
        """
            Extract brand using contextual indicators like 'by', 'from', 'of'
            
            Args:
                text (str): Input text to analyze
                
            Returns:
                Optional[Tuple[str, float]]: Tuple of (brand_name, confidence_score) or None
        """
        # First, try with original text
        text_lower = text.lower()
        
        # Also try with spell-corrected text
        corrected_text = self._spellCorrectText(text)
        corrected_lower = corrected_text.lower()
        
        # Try both original and corrected versions
        for text_version in [text_lower, corrected_lower]:
            for indicator in self.brand_indicators:
                # Pattern to capture brand name after indicator
                escaped_indicators = "|".join(map(re.escape, self.product_indicators))
                pattern = rf'{re.escape(indicator)}\s+([a-zA-Z0-9\s\-&\.]+?)(?:\s*[,;!?.]|\s+(?:{escaped_indicators})|$)'

                matches = re.finditer(pattern, text_version, re.IGNORECASE)
                
                for match in matches:
                    potential_brand = match.group(1).strip()
                    
                    # Skip if empty or too short
                    if not potential_brand or len(potential_brand) < 2:
                        continue
                    
                    # Check if it's an exact match (case-insensitive)
                    if potential_brand in self.brand_names:
                        return (self.brand_names[potential_brand], 1.0)
                    
                    # Try fuzzy matching
                    fuzzy_result = fuzzyMatch(potential_brand, self.brand_names, threshold=0.85)
                    if fuzzy_result:
                        return fuzzy_result
        
        return None
    
    def _extractBrandDirect(self, text: str) -> Optional[Tuple[str, float]]:
        """
            Extract brand by direct matching anywhere in text
            FIXED: Prevents false positives from partial word matches
            
            Args:
                text (str): Input text to analyze
                
            Returns:
                Optional[Tuple[str, float]]: Tuple of (brand_name, confidence_score) or None
        """
        text_lower = text.lower()
        words_in_query = set(text_lower.split())
        
        # Check for common descriptors that shouldn't match brands
        for word in words_in_query:
            if self._isCommonDescriptor(word):
                logger.debug(f"Skipping common descriptor: {word}")
        
        # Try to find exact brand name matches with proper word boundaries
        for brand_lower, brand_original in self.brand_names.items():
            # CRITICAL FIX: Only match complete words, not partial matches
            # Use word boundary regex to ensure we're matching complete words
            pattern = rf'\b{re.escape(brand_lower)}\b'
            
            if re.search(pattern, text_lower):
                # Additional validation: Check if this is actually a brand mention
                # or just a coincidental substring match
                
                # If the brand name is very short (< 4 chars), require stronger evidence
                if len(brand_lower) < 4:
                    # For short brand names, require exact word match or brand indicator
                    if brand_lower not in words_in_query:
                        continue
                    
                    # Check if any brand indicator is present
                    has_indicator = any(indicator in text_lower for indicator in self.brand_indicators)
                    if not has_indicator:
                        # Skip if it looks like a common word
                        if self._isCommonDescriptor(brand_lower):
                            continue
                
                # Check if the matched brand is actually part of a common descriptor phrase
                # For example, "mid" in "mid century" shouldn't match brand "MIDJ"
                match_start = text_lower.find(brand_lower)
                if match_start >= 0:
                    # Get context around the match
                    context_start = max(0, match_start - 10)
                    context_end = min(len(text_lower), match_start + len(brand_lower) + 10)
                    context = text_lower[context_start:context_end]
                    
                    # Check if this is part of a common phrase
                    skip_phrases = ['mid century', 'inclined', 'platform', 'with platform']
                    should_skip = False
                    for phrase in skip_phrases:
                        if brand_lower in phrase and phrase in text_lower:
                            # The brand match is actually part of a common phrase
                            logger.debug(f"Skipping false positive: '{brand_lower}' is part of '{phrase}'")
                            should_skip = True
                            break
                    
                    if should_skip:
                        continue
                
                logger.info(f"Found brand match: '{brand_original}' in '{text}'")
                return (brand_original, 1.0)
        
        return None
    
    def _extractProductWithWindow(self, text: str, brand_name: Optional[str] = None) -> Optional[Tuple[str, float]]:
        """
            Extract product name using sliding window approach with spell correction
            
            Args:
                text (str): Input query text
                brand_name (str, optional): Brand name to exclude from matching
                
            Returns:
                Optional[Tuple[str, float]]: Tuple of (product_name, confidence_score) or None
        """
        # Try both original and spell-corrected text
        original_lower = text.lower()
        corrected_text = self._spellCorrectText(text)
        corrected_lower = corrected_text.lower()
        
        best_overall_match = None
        best_overall_score = 0
        
        for text_version in [original_lower, corrected_lower]:
            text_to_process = text_version
            
            # Remove brand name from text if found
            if brand_name:
                text_to_process = re.sub(rf'\b{re.escape(brand_name.lower())}\b', '', text_to_process)
            
            # Remove brand indicators and common words
            removal_patterns = self.brand_indicators + self.product_indicators + \
                              ['i', 'want', 'need', 'looking', 'for', 'the', 'a', 'an']
            for pattern in removal_patterns:
                text_to_process = re.sub(rf'\b{re.escape(pattern)}\b', '', text_to_process)
            
            # Clean up extra spaces
            text_to_process = ' '.join(text_to_process.split())
            
            if not text_to_process:
                continue
            
            # Try different window sizes for product name matching (largest first)
            words = text_to_process.split()
            
            for window_size in range(min(5, len(words)), 0, -1):
                for i in range(len(words) - window_size + 1):
                    phrase = ' '.join(words[i:i + window_size])
                    
                    # Skip very short phrases
                    if len(phrase) < 2:
                        continue
                    
                    # Check for exact match
                    if phrase in self.product_names:
                        return (self.product_names[phrase], 1.0)
                    
                    # Try fuzzy matching
                    fuzzy_result = fuzzyMatch(phrase, self.product_names, 
                                                    threshold=self.loose_threshold)
                    if fuzzy_result and fuzzy_result[1] > best_overall_score:
                        best_overall_match = fuzzy_result[0]
                        best_overall_score = fuzzy_result[1]
        
        return (best_overall_match, best_overall_score) if best_overall_match else None
    
    def _extractBrandWithPreprocessing(self, text: str) -> Optional[Tuple[str, float]]:
        """
            Extract brand with preprocessing to handle typos in multi-word brands
            FIXED: Added validation to prevent false positives
            
            Args:
                text (str): Input text to analyze
                
            Returns:
                Optional[Tuple[str, float]]: Tuple of (brand_name, confidence_score) or None
        """
        text_lower = text.lower()
        
        # First, try with indicators (most reliable)
        brand_with_indicators = self._extractBrandWithIndicators(text)
        if brand_with_indicators:
            return brand_with_indicators
        
        # Second, try direct matching (with improved validation)
        direct_match = self._extractBrandDirect(text)
        if direct_match:
            return direct_match
        
        # Third, try fuzzy matching but with stricter thresholds for descriptive words
        best_match = None
        best_score = 0
        
        for brand_lower, brand_original in self.brand_names.items():
            # Skip if brand name is too similar to common descriptors
            if self._isCommonDescriptor(brand_lower):
                continue
            
            # Split brand into words
            brand_words = brand_lower.split()
            query_words = text_lower.split()
            
            # Skip if any query word is a common descriptor that partially matches brand
            skip_brand = False
            for query_word in query_words:
                if self._isCommonDescriptor(query_word):
                    # Check if this descriptor is too similar to the brand name
                    similarity = self._calculateSimilarity(query_word, brand_lower)
                    if 0.4 < similarity < 0.9:  # Partial match that's not exact
                        skip_brand = True
                        break
            
            if skip_brand:
                continue
            
            # Try to find all brand words in query (allowing for typos)
            found_positions = []
            
            for brand_word in brand_words:
                best_word_match = None
                best_word_score = 0
                best_word_pos = -1
                
                for i, query_word in enumerate(query_words):
                    # Skip common descriptors
                    if self._isCommonDescriptor(query_word):
                        continue
                    
                    # Calculate similarity
                    similarity = self._calculateSimilarity(brand_word, query_word)
                    
                    if similarity > best_word_score:
                        best_word_score = similarity
                        best_word_match = query_word
                        best_word_pos = i
                
                # Require higher threshold for matching
                if best_word_score >= 0.85:  # Increased from 0.75
                    found_positions.append((best_word_pos, best_word_score))
            
            # If we found all brand words with good similarity
            if len(found_positions) == len(brand_words):
                # Check if they appear in sequence (or close together)
                positions = [pos for pos, _ in found_positions]
                scores = [score for _, score in found_positions]
                
                # Words should be within 3 positions of each other
                if max(positions) - min(positions) <= len(brand_words) + 2:
                    avg_score = sum(scores) / len(scores)
                    
                    if avg_score > best_score:
                        best_score = avg_score
                        best_match = brand_original
                        logger.info(f"Fuzzy brand match: '{text}' -> '{brand_original}' (confidence: {avg_score:.2f})")
        
        # Require higher threshold for fuzzy matches
        if best_match and best_score >= 0.85:  # Increased from 0.75
            return (best_match, best_score)
        
        return None
    
    def extractProductBrand(self, text: str) -> Dict[str, Optional[Dict[str, any]]]:
        """
            Extract both brand and product from query text
            
            Args:
                text (str): Input query text
                
            Returns:
                Dict containing:
                    - brand: Dict with 'name' and 'confidence' keys or None
                    - product: Dict with 'name' and 'confidence' keys or None
        """
        result = {
            'brand': None,
            'product': None
        }
        logger.info(f'Processing query: "{text}" | Loaded brands: {len(self.brand_names)}, products: {len(self.product_names)}')
        
        # Extract brand with improved validation
        brand_result = self._extractBrandWithPreprocessing(text)
        
        if brand_result:
            result['brand'] = {
                'name': brand_result[0],
                'confidence': brand_result[1]
            }
            logger.info(f"Extracted brand: {brand_result[0]} (confidence: {brand_result[1]:.2f})")
        else:
            logger.info(f"No brand found in: {text}")
        
        # Extract product (exclude brand name from search)
        brand_name = brand_result[0] if brand_result else None
        product_result = self._extractProductWithWindow(text, brand_name)
        
        if product_result:
            result['product'] = {
                'name': product_result[0],
                'confidence': product_result[1]
            }
            logger.info(f"Extracted product: {product_result[0]} (confidence: {product_result[1]:.2f})")
        else:
            logger.info(f"No product found in: {text}")
        
        return result