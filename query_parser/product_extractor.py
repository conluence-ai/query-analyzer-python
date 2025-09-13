# Import necessary libraries
import re
import torch
import logging
from typing import Dict, List, Tuple
from functools import lru_cache
from config.constants import FURNITURE_TYPE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)


class ProductExtractor:
    """Advanced NER using spaCy and BERT"""
    
    # def __init__(self, model_name: str = "en_core_web_sm"):
    def __init__(self):
        """
            Initialize the NER model
            
            Args:
                model_name (str): Name of the spaCy model to load
        """
        # Zero-shot classifier for product types
        self._zero_shot_model = None
        self.confidence_threshold = 0.3
        self._preparePatterns()
        self._classifier_loaded = False

    def _preparePatterns(self):
        """Pre-compile regex patterns for faster matching"""
        self.category_patterns = {}
        self.term_to_category = {}
            
        for main_category, synonyms in FURNITURE_TYPE.items():
            all_terms = [main_category.lower()] + [syn.lower() for syn in synonyms]
            patterns = []
                
            for term in all_terms:
                # Create regex pattern for word boundaries
                pattern = re.compile(r'\b' + re.escape(term) + r's?\b', re.IGNORECASE)
                patterns.append((term, pattern, main_category))
                self.term_to_category[term] = main_category
                
            self.category_patterns[main_category] = patterns

    @lru_cache(maxsize=1000)
    def classifyProductType(self, text: str) -> Tuple[List[str], List[float]]:
        """
        Optimized product type classification with caching
        
        Args:
            text (str): Input text to classify
            
        Returns:
            Tuple[List[str], List[float]]: Predicted labels and confidence scores
        """
        if not text or not text.strip():
            return ["Unknown"], [0.0]
        
        try:
            text_lower = text.lower().strip()
            
            # Step 1: Fast keyword matching with pre-compiled patterns
            category_matches = self._matchProductKeywords(text_lower)
            
            if category_matches:
                # Sort by confidence (descending)
                sorted_matches = sorted(category_matches.items(), key=lambda x: x[1][0], reverse=True)
                categories = [item[0] for item in sorted_matches]
                confidences = [item[1][0] for item in sorted_matches]
                return categories, confidences
            
            # Step 2: Fallback to zero-shot only if no keyword matches
            return self._zeroShotClassification(text, text_lower)
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return ["Unknown"], [0.0]
        
    def _matchProductKeywords(self, text_lower: str) -> Dict:
        """Optimized keyword matching using pre-compiled patterns"""
        category_matches = {}
        
        for main_category, patterns in self.category_patterns.items():
            matched_terms = []
            max_confidence = 0
            
            for term, pattern, category in patterns:
                if pattern.search(text_lower):
                    matched_terms.append(term)
                    # Higher confidence for exact main category matches
                    conf = 0.95 if term == main_category.lower() else 0.85
                    max_confidence = max(max_confidence, conf)
            
            if matched_terms:
                category_matches[main_category] = (max_confidence, matched_terms)
        
        return category_matches

    @property
    def classificationPipeline(self):
        """Lazy load the zero-shot classifier only when needed"""
        if not self._classifier_loaded:
            try:
                from transformers import pipeline
                self._zero_shot_model = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    device=0 if self._hasGpu() else -1  # Use GPU if available
                )
                logger.info("Zero-shot classifier loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load zero-shot classifier: {e}")
                self._zero_shot_model = None
            finally:
                self._classifier_loaded = True
        return self._zero_shot_model
    
    def _hasGpu(self):
        """Check if GPU is available"""
        try:
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _zeroShotClassification(self, original_text: str, text_lower: str) -> Tuple[List[str], List[float]]:
        """Fallback zero-shot classification with optimizations"""
        if not self.classificationPipeline:
            return ["Unknown"], [0.0]
        
        try:
            # Use only main categories for faster inference
            main_categories = list(FURNITURE_TYPE.keys())
            
            # Limit text length for faster processing
            truncated_text = original_text[:200] if len(original_text) > 200 else original_text
            
            result = self.classificationPipeline(truncated_text, main_categories)
            
            # Process results
            high_confidence_results = []
            for label, score in zip(result['labels'], result['scores']):
                if score >= self.confidence_threshold:
                    high_confidence_results.append((label, score))
            
            if high_confidence_results:
                categories = [item[0] for item in high_confidence_results]
                confidences = [item[1] for item in high_confidence_results]
                return categories, confidences
            else:
                # Return top result even if below threshold
                return [result['labels'][0]], [result['scores'][0]]
                
        except Exception as e:
            logger.error(f"Zero-shot classification error: {e}")
            return ["Unknown"], [0.0]