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
from query_parser.product_extractor import ProductExtractor
from query_parser.price_extractor import PriceExtractor
from query_parser.style_extractor import StyleExtractor
from query_parser.classification_extractor import StyleClassification

# Import constants and mappings
from config.constants import FURNITURE_CATEGORY

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
        self.product_extractor = ProductExtractor()
        self.price_extractor = PriceExtractor()
        self.style_extractor = StyleExtractor()
        self.classification_extractor = StyleClassification()

        # Fuzzy matching threshold
        self.fuzzy_threshold = 0.95
        self.all_synonyms = set()  # For fuzzy matching

        # Spell checker (optional, lightweight)
        self.spell = SpellChecker(distance=2) 

        # Feature dictionary and mappings
        self.synonym_to_feature = {}
        self.feature_to_category = {}
        
        # Category mappings for better organization
        self.category_mappings = {
            'Metal Legs': 'Legs',
            'Straight Legs': 'Legs', 
            'Planar Legs': 'Legs',
            'Wooden plinth': 'Base',
            'Metal plinth': 'Base',
            'upholstered base': 'Base',
            'Heavy Base': 'Base',
            'Upholstered Plinth': 'Base',
            'With Armrest': 'Arms',
            'Without Armrest': 'Arms',
            'Squared Arms': 'Arms',
            'roundedArms': 'Arms',
            'Rounded Arms': 'Arms',
            'Flat Arms': 'Arms',
            'Flared arms': 'Arms',
            'Sinuous Curve Arms': 'Arms',
            'Sloping Arms': 'Arms',
            'Folded Arms': 'Arms',
            'Sleek Arms': 'Arms',
            'Splayed Arms': 'Arms',
            'Floor Length Arms': 'Arms',
            'Modular Arm': 'Arms',
            'armsWithMetalDetail': 'Arms',
            'Armrest Height Aligned with Backrest': 'Arms',
            'Armrest integrated with Structure': 'Arms',
            'Fixed Arms': 'Arms',
            'Low Back': 'Back',
            'Mid Back': 'Back',
            'Flared High Back': 'Back',
            'LowBack': 'Back',
            'Plain Back without division': 'Back',
            'Piping on Back': 'Back',
            'Grid Tufted Back': 'Back',
            'Cylindrical Back': 'Back',
            'Angular Back': 'Back',
            'Pleated Back': 'Back',
            'Quilted Back': 'Back',
            'Adjustable Back': 'Back',
            'BackCushionsIntegral': 'Back',
            'Leather Back Covering': 'Back',
            'Split Seat': 'Seat',
            'Non-Uniform Seat Division': 'Seat',
            'Fabric': 'Material',
            'Leather Piping': 'Details',
            'Braid Piping': 'Details',
            'Flat Piping': 'Details',
            'Piping Follows Structure': 'Details',
            'Metal detail': 'Details',
            'Metal Structure': 'Structure',
            'Metal Wire Frame': 'Structure',
            'Continuous Structure': 'Structure',
            'upholstered structure': 'Structure',
            'Tubular Hollow Wooden Frame': 'Structure',
            'Integrated Arms & Legs': 'Structure',
            'Upholstered Shell': 'Structure',
            'Curved': 'Shape',
            'Organic Shape': 'Shape',
            'Sinuous': 'Shape',
            'LShape': 'Shape',
            'Bean': 'Shape',
            'Chesterfield': 'Style',
            'Horizontal Tufting': 'Style',
            'WithOptionalLooseCushions': 'Accessories',
            'Wooden Legs': 'Legs',
        }

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
            category = self.category_mappings.get(main_feature, 'Other')
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
        
        # Define contextual patterns that might not be exact matches
        contextual_patterns = [
            # Leg patterns
            (r'\b(wooden|wood)\s+(leg|legs)\b', 'Wooden Legs'),
            (r'\b(metal|steel|chrome)\s+(leg|legs)\b', 'Metal Legs'),
            (r'\b(straight|vertical)\s+(leg|legs)\b', 'Straight Legs'),
            
            # Arm patterns
            (r'\b(no|without|armless)\s+(arm|arms)\b', 'Without Armrest'),
            (r'\b(with|having)\s+(arm|arms|armrest)\b', 'With Armrest'),
            (r'\b(rounded|curved|circular)\s+(arm|arms)\b', 'Rounded Arms'),
            (r'\b(square|squared|rectangular)\s+(arm|arms)\b', 'Squared Arms'),
            (r'\b(flat|slab)\s+(arm|arms)\b', 'Flat Arms'),
            
            # Back patterns
            (r'\b(low|short)\s+(back|backrest)\b', 'Low Back'),
            (r'\b(high|tall)\s+(back|backrest)\b', 'Flared High Back'),
            (r'\b(curved|cylindrical|round)\s+(back|backrest)\b', 'Cylindrical Back'),
            
            # Material patterns
            (r'\b(leather|genuine\s+leather|real\s+leather)\b', 'Leather'),
            (r'\b(fabric|cloth|textile|upholstery)\b', 'Fabric'),
            
            # Structure patterns
            (r'\b(metal|steel|iron)\s+(frame|structure)\b', 'Metal Structure'),
            (r'\b(wire|wireframe)\s+(frame|structure)\b', 'Metal Wire Frame'),
            
            # Shape patterns
            (r'\b(l[\s-]?shaped|corner)\b', 'LShape'),
            (r'\b(curved|organic|flowing)\b', 'Curved'),
            
            # Style patterns
            (r'\b(chesterfield|tufted|buttoned)\b', 'Chesterfield'),
            (r'\b(bean|bean[\s-]?bag)\b', 'Bean'),
        ]
        
        # Fuzzy pattern matching for common misspellings
        fuzzy_patterns = [
            # Common misspellings for materials
            (r'\b(lether|leater|leathr)\b', 'Leather'),
            (r'\b(fabrik|febric|fabic)\b', 'Fabric'),
            (r'\b(mettal|metel|matel)\b', 'Metal'),
            
            # Common misspellings for furniture parts
            (r'\b(cussion|cushon|cushin)\b', 'cushion'),
            (r'\b(armrest|armrst|armest)\b', 'armrest'),
        ]
        
        # Apply regular patterns
        for pattern, feature in contextual_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                if feature not in detected:
                    detected.append(feature)
        
        # Apply fuzzy patterns
        for pattern, corrected_word in fuzzy_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Replace the misspelled word and recheck patterns
                corrected_text = text.replace(match.group(), corrected_word)
                for context_pattern, feature in contextual_patterns:
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
        """
            Enhanced parsing with ML models

            Args:
                query (str): Input text to parse

            Returns:
                ParserResult: Parsed result with product type, features, price range, location, and confidence    
        """
        logger.info(f"ML parsing query: {query}")
        
        # Classify product type using zero-shot
        product_types, product_confidence = self.product_extractor.classifyProductType(query)

        # Extract features using hybrid approach
        features = self._extractFeatures(query)

        # Extract styles using hybrid approach
        styles = self.style_extractor.extractStyles(query)

        # Extract styles using hybrid approach
        classfications = self.classification_extractor.extractClassification(query)

        # Extract price range using ML model
        price_range = self.price_extractor.extractPriceRange(query)

        result = ParserResult(
            product_type=product_types if product_types != ["Unknown"] else [],
            brand_name=[],
            product_name=[],
            features=features,
            price_range=price_range,
            location="",
            confidence_score=0,
            styles=styles,
            classification_summary=classfications,
            extras=[],
            original_query=query
        )
        
        return result
    
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