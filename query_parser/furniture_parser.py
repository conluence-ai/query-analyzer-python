"""Furniture Parser using ML/NLP models and hybrid techniques"""

# Import necessary libraries
import re
import logging
from typing import Dict, List, Set, Optional
from config.config import ParserResult
from difflib import SequenceMatcher
import Levenshtein  # pip install python-Levenshtein
from spellchecker import SpellChecker  # pip install pyspellchecker

# Import custom modules
from query_parser.product_type_extractor import ProductTypeExtractor
from query_parser.price_extractor import PriceExtractor
from query_parser.style_extractor import StyleExtractor
from query_parser.classification_extractor import StyleClassification
from query_parser.brand_product_extraction import BrandProductExtractor

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

class FurnitureParser:
    """Furniture parser with ML/NLP models"""
    
    def __init__(self):
        """
            Initialize the furniture parser
        """
        # Load Extraction models
        self.product_type_extractor = ProductTypeExtractor()
        self.price_extractor = PriceExtractor()
        self.style_extractor = StyleExtractor()
        self.classification_extractor = StyleClassification()
        self.brand_product_extractor = BrandProductExtractor()

        # Fuzzy matching threshold
        self.fuzzy_threshold = 0.95
        self.all_synonyms = set()  # For fuzzy matching

        # Spell checker (optional, lightweight)
        self.spell = SpellChecker(distance=2) 

        # Feature dictionary and mappings
        self.synonym_to_feature = {}
        self.feature_to_category = {}
        
        # Build mappings from your FURNITURE_CATEGORY
        self._buildFeatureMappings()

    def _buildFeatureMappings(self):
        """Build reverse mapping from synonyms to main features"""
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

    def _fuzzy_match(self, phrase: str, candidates: Set[str]) -> Optional[str]:
        """Fuzzy match entire phrase against known synonyms"""
        best_match, best_score = None, 0

        for candidate in candidates:
            try:
                similarity = Levenshtein.ratio(phrase.lower(), candidate.lower())
            except (ImportError, NameError):
                similarity = SequenceMatcher(None, phrase.lower(), candidate.lower()).ratio()

            if similarity > best_score and similarity >= self.fuzzy_threshold:
                best_score, best_match = similarity, candidate

        return best_match
    
    def _spell_correct(self, text: str) -> str:
        """Optional spell correction before fuzzy matching"""
        corrected = []
        for w in text.split():
            if len(w) > 2:  # avoid correcting very short tokens
                corrected.append(self.spell.correction(w) or w)
            else:
                corrected.append(w)
        return " ".join(corrected)

    def getCategoriesFromText(self, text: str) -> List[str]:
        """Extract features by direct keyword matching and fuzzy matching"""
        detected = []
        words = text.split()

        # Step 0: Spell correction
        corrected_text = self._spell_correct(text)
        corrected_words = corrected_text.split()
        
        # First pass: Exact match
        for window_size in [1, 2, 3]:
            for i in range(len(corrected_words) - window_size + 1):
                phrase = ' '.join(corrected_words[i:i + window_size])
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
                    fuzzy_match = self._fuzzy_match(phrase, self.all_synonyms)
                    if fuzzy_match:
                        main_feature = self.synonym_to_feature[fuzzy_match]
                        if self._hasContextualMatch(phrase, corrected_words, i):
                            if main_feature not in detected:
                                detected.append(main_feature)
        
        return detected
    
    def _hasContextualMatch(self, phrase: str, words: List[str], position: int) -> bool:
        """Check if feature is contextually relevant"""
        if "leather" in phrase:
            context_window = words[max(0, position - 2):min(len(words), position + 3)]
            furniture_parts = ['sofa', 'chair', 'back', 'seat', 'arm', 'cushion']
            return any(part in ' '.join(context_window).lower() for part in furniture_parts)
        return True
    
    def extractContextualCategories(self, text: str) -> List[str]:
        """Extract features using contextual phrase matching with fuzzy support"""
        detected = []
        
        # Apply regular patterns
        for pattern, feature in FEATURE_CONTEXTUAL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                if feature not in detected:
                    detected.append(feature)
        
        # Apply fuzzy patterns
        for pattern, corrected_word in FUZZY_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Replace the misspelled word and recheck patterns
                corrected_text = text.replace(match.group(), corrected_word)
                for context_pattern, feature in FEATURE_CONTEXTUAL_PATTERNS:
                    if re.search(context_pattern, corrected_text, re.IGNORECASE):
                        if feature not in detected:
                            detected.append(feature)
        
        return detected
    
    def _extractFeatures(self, text: str) -> List[str]:
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
        keyword_features = self.getCategoriesFromText(text_lower)
        detected_features.extend(keyword_features)
        
        # Method 2: Contextual phrase matching
        contextual_features = self.extractContextualCategories(text_lower)
        detected_features.extend(contextual_features)
        
        # Remove duplicates while preserving order
        unique_features = []
        seen = set()
        for feature in detected_features:
            if feature not in seen:
                unique_features.append(feature)
                seen.add(feature)
        
        return unique_features
    
    def structureQuery(self, query: str) -> ParserResult:
        """Enhanced parsing with ML models"""
        logger.info(f"ML parsing query: {query}")
        
        # STEP 1: Extract brand/product FIRST (before anything else)
        data = self.brand_product_extractor.extract(query)
        
        # Safely extract brand name
        brand_name = data['brand']['name'] if data and data.get('brand') else None
        logger.info(f"Extracted brand: {brand_name}")
        
        # Verify and extract product name
        product_name = None
        if data and data.get('product'):
            extracted_product = data['product']['name']
            if self._verify_product_in_query(extracted_product, query):
                product_name = extracted_product
                logger.info(f"Verified product: {product_name}")
            else:
                logger.info(f"Product '{extracted_product}' not verified in query")
        
        # STEP 2: Create cleaned query for price extraction
        # Remove brand and product names to prevent number confusion
        cleaned_query_for_price = query
        
        if brand_name:
            # Remove brand name (case-insensitive)
            cleaned_query_for_price = re.sub(
                rf'\b{re.escape(brand_name)}\b', 
                '', 
                cleaned_query_for_price, 
                flags=re.IGNORECASE
            )
            logger.info(f"Cleaned query after removing brand: {cleaned_query_for_price}")
        
        if product_name:
            # Remove product name (case-insensitive)
            cleaned_query_for_price = re.sub(
                rf'\b{re.escape(product_name)}\b', 
                '', 
                cleaned_query_for_price, 
                flags=re.IGNORECASE
            )
            logger.info(f"Cleaned query after removing product: {cleaned_query_for_price}")
        
        # STEP 3: Extract other features from ORIGINAL query
        product_types, product_confidence = self.product_type_extractor.classifyProductType(query)
        features = self._extractFeatures(query)
        styles = self.style_extractor.extractStyles(query)
        classifications = self.classification_extractor.extractClassification(query)
        
        # STEP 4: Extract price from CLEANED query only
        price_range = self.price_extractor.extractPriceRange(cleaned_query_for_price)
        
        # STEP 5: Validate price range
        # If price was extracted and matches a number in the original brand/product name, invalidate it
        if price_range and price_range.max:
            should_invalidate = False
            
            # Check if price matches any number in brand name
            if brand_name and any(char.isdigit() for char in brand_name):
                brand_numbers = re.findall(r'\d+', brand_name)
                for num_str in brand_numbers:
                    if float(num_str) == price_range.max or float(num_str) == price_range.min:
                        logger.warning(f"Price {price_range.max or price_range.min} matches brand number '{num_str}', invalidating")
                        should_invalidate = True
                        break
            
            # Check if price matches any number in product name
            if product_name and any(char.isdigit() for char in product_name):
                product_numbers = re.findall(r'\d+', product_name)
                for num_str in product_numbers:
                    if float(num_str) == price_range.max or float(num_str) == price_range.min:
                        logger.warning(f"Price {price_range.max or price_range.min} matches product number '{num_str}', invalidating")
                        should_invalidate = True
                        break
            
            if should_invalidate:
                price_range = None

        result = ParserResult(
            product_type=product_types if product_types != ["Unknown"] else [],
            brand_name=brand_name,
            product_name=product_name,
            features=features,
            price_range=price_range,
            location="",
            confidence_score=product_confidence[0] if product_confidence else 0.0,
            styles=styles,
            classification_summary=classifications,
            extras=[],
            original_query=query
        )
        
        return result
    
    def _verify_product_in_query(self, product_name: str, query: str) -> bool:
        """
        Verify if a product name is actually mentioned in the query
        
        Args:
            product_name: The extracted product name
            query: The original query text
            
        Returns:
            bool: True if product name appears in query
        """
        if not product_name:
            return False
        
        # Normalize both strings for comparison
        query_lower = query.lower()
        product_lower = product_name.lower()
        
        # Check if the full product name appears
        if product_lower in query_lower:
            return True
        
        # Check if significant words from product name appear (at least 70% of words)
        product_words = set(product_lower.split())
        query_words = set(query_lower.split())
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'at'}
        product_words = product_words - common_words
        
        if not product_words:
            return False
        
        matching_words = product_words & query_words
        match_ratio = len(matching_words) / len(product_words)
        
        return match_ratio >= 0.7  # At least 70% of product name words must appear
    
    def analyzeQueryText(self, query: str) -> ParserResult:
        """
            Parse query and return as dictionary

            Args:
                query (str): Input text to parse

            Returns:
                ParserResult: Parsed result with product type, features, price range, location, and confidence    
        """
        result = self.structureQuery(query)
        
        return {
            "product_type": result.product_type, 
            "features": result.features,
            "brand_name": result.brand_name,
            "product_name": result.product_name,
            "styles": result.styles,
            "price_range": result.price_range,
            "location": result.location,
            "classification_summary": result.classification_summary,
            "extras": result.extras,
            "confidence_score": result.confidence_score,
            "original_query": result.original_query
        }