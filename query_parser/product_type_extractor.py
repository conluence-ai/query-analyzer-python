""" Product Type Extractor for furniture queries """

# Import necessary libraries
import re
import logging
import numpy as np
from functools import lru_cache
from typing import Dict, List, Tuple, Set
from difflib import SequenceMatcher

# Import constants and mappings
from config.constants import FURNITURE_TYPE, DISAMBIGUATION_RULES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class ProductTypeExtractor:
    """ Extract furniture type from text using fuzzy matching and spell correction """

    def __init__(self):
        """ Initialize the style extractor """

        # Fuzzy and vector matching threshold
        self.confidence_threshold = 0.7
        
        # Fuzzy matching threshold
        self.fuzzy_threshold = 0.8  # for fuzzy matching

        self._preparePatterns()
        self._buildVectorSearch()
        self._handleAmbiguousTerms()
        logger.info("Lightweight furniture product extractor initialized")

    def _handleAmbiguousTerms(self):
        """ Handle terms that appear in multiple categories """
        self.ambiguous_terms = {}
        
        # Find terms that appear in multiple categories
        term_categories = {}
        for main_cat, synonyms in FURNITURE_TYPE.items():
            all_terms = [main_cat.lower()] + [syn.lower() for syn in synonyms]
            for term in all_terms:
                if term not in term_categories:
                    term_categories[term] = []
                term_categories[term].append(main_cat)
        
        # Store ambiguous terms with priority rules
        for term, categories in term_categories.items():
            if len(categories) > 1:
                self.ambiguous_terms[term] = categories

    def _preparePatterns(self):
        """ Build regex patterns and term mappings """
        self.category_patterns = {}
        self.term_to_category = {}
        self.all_terms = set()
        
        for main_category, synonyms in FURNITURE_TYPE.items():
            all_terms = [main_category.lower()] + [syn.lower() for syn in synonyms]
            patterns = []
            
            for term in all_terms:
                # Add term variations (plural, etc.)
                term_variants = [term, term + 's', term.rstrip('s')]
                for variant in term_variants:
                    if len(variant) > 2:  # avoid very short terms
                        pattern = re.compile(r'\b' + re.escape(variant) + r'\b', re.IGNORECASE)
                        patterns.append((variant, pattern, main_category))
                        self.term_to_category[variant] = main_category
                        self.all_terms.add(variant)
            
            self.category_patterns[main_category] = patterns

    def _buildVectorSearch(self):
        """ Build simple character-based vectors for fuzzy matching """
        self.term_vectors = {}
        self.term_list = list(self.all_terms)
        
        # Create character frequency vectors for each term
        for term in self.term_list:
            self.term_vectors[term] = self._getCharVector(term)

    def _getCharVector(self, text: str, max_chars: int = 26) -> np.ndarray:
        """
            Creates a simple character frequency vector (histogram) from the input text.

            Args:
                text (str): The input string to convert into a character frequency vector.
                max_chars (int, optional): The length of the resulting vector, assumed 
                                        to be 26 (for 'a' to 'z'). Defaults to 26.

            Returns:
                np.ndarray: A NumPy array representing the normalized character 
                            frequency distribution of the text.
        """
        text = text.lower()
        vector = np.zeros(max_chars)
        
        for char in text:
            if 'a' <= char <= 'z':
                vector[ord(char) - ord('a')] += 1
        
        # Normalize
        if np.sum(vector) > 0:
            vector = vector / np.sum(vector)
        
        return vector

    def _cosineSimilarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
            Calculates the cosine similarity between two NumPy vectors.

            Args:
                vec1 (np.ndarray): The first vector (NumPy array).
                vec2 (np.ndarray): The second vector (NumPy array).

            Returns:
                float: The calculated cosine similarity score.
        """
        dot_product = np.dot(vec1, vec2)
        norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        if norms == 0:
            return 0.0
        
        return dot_product / norms

    def _exactMatch(self, text: str) -> Tuple[List[str], List[float]]:
        """
            Performs exact classification of product types by matching the input text

            Args:
                text (str): The cleaned, lowercased input string (e.g., a search query).

            Returns:
                Tuple[List[str], List[float]]: A tuple containing a list of matching standardized 
                            category names and a corresponding list of confidence scores. 
                            Returns `None` if no exact match is found.
        """
        category_matches = {}
        
        for main_category, patterns in self.category_patterns.items():
            max_conf = 0
            matched_terms = []
            
            for term, pattern, category in patterns:
                if pattern.search(text):
                    matched_terms.append(term)
                    
                    # Handle disambiguation for ambiguous terms
                    if hasattr(self, 'disambiguation_rules') and term in DISAMBIGUATION_RULES:
                        priority_cats = DISAMBIGUATION_RULES[term]
                        if category in priority_cats:
                            # Higher confidence for prioritized categories
                            priority_index = priority_cats.index(category)
                            conf = 0.98 - (priority_index * 0.02)  # 0.98, 0.96, 0.94...
                        else:
                            conf = 0.85
                    else:
                        # Higher confidence for exact main category match
                        conf = 0.98 if term == main_category.lower() else 0.95
                    
                    max_conf = max(max_conf, conf)
            
            if matched_terms:
                category_matches[main_category] = max_conf

        if category_matches:
            sorted_matches = sorted(category_matches.items(), key=lambda x: x[1], reverse=True)
            categories = [item[0] for item in sorted_matches]
            confidences = [item[1] for item in sorted_matches]
            return categories, confidences
        
        return None

    def _fuzzyMatch(self, text: str) -> Tuple[List[str], List[float]]:
        """
            Find the best fuzzy match for a phrase among candidates
                
            Args:
                phrase (str): The phrase to match
                candidates (Dict[str, str]): Dictionary mapping lowercase names to original names
                threshold (float, optional): Similarity threshold (0-1)
                    
            Returns:
                Optional[Tuple[List[str], List[float]]]: Tuple of ([matched_name], [similarity_score]) or None
        """
        words = re.findall(r'\b\w{3,}\b', text)  # Extract words 3+ chars
        if not words:
            return None

        category_scores = {}
        
        for word in words:
            for term in self.all_terms:
                if len(term) >= 3:
                    # Use SequenceMatcher for fuzzy matching
                    similarity = SequenceMatcher(None, word, term).ratio()
                    
                    if similarity >= self.fuzzy_threshold:
                        main_cat = self.term_to_category[term]
                        # Adjust confidence based on similarity
                        confidence = similarity * 0.9  # max 0.9 for fuzzy
                        
                        if main_cat not in category_scores or category_scores[main_cat] < confidence:
                            category_scores[main_cat] = confidence

        if category_scores:
            # Filter by confidence threshold
            filtered_scores = {k: v for k, v in category_scores.items() 
                             if v >= self.confidence_threshold}
            
            if filtered_scores:
                sorted_matches = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)
                categories = [item[0] for item in sorted_matches]
                confidences = [item[1] for item in sorted_matches]
                return categories, confidences
        
        return None

    def _vectorMatch(self, text: str) -> Tuple[List[str], List[float]]:
        """
            Performs product classification using character vector similarity matching.

            Args:
                text (str): The cleaned input string to be classified.

            Returns:
                Tuple[List[str], List[float]]: A tuple containing a list of matching standardized 
                            category names and a corresponding list of confidence scores. 
                            Returns `None` if no confident match is found.
        """
        words = re.findall(r'\b\w{3,}\b', text)
        if not words:
            return None

        category_scores = {}
        
        for word in words:
            word_vector = self._getCharVector(word)
            
            for term in self.all_terms:
                if len(term) >= 3:
                    term_vector = self.term_vectors[term]
                    similarity = self._cosineSimilarity(word_vector, term_vector)
                    
                    if similarity >= 0.7:  # Vector similarity threshold
                        main_cat = self.term_to_category[term]
                        confidence = similarity * 0.85  # max 0.85 for vector
                        
                        if main_cat not in category_scores or category_scores[main_cat] < confidence:
                            category_scores[main_cat] = confidence

        if category_scores:
            filtered_scores = {k: v for k, v in category_scores.items() 
                             if v >= self.confidence_threshold}
            
            if filtered_scores:
                sorted_matches = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)
                categories = [item[0] for item in sorted_matches]
                confidences = [item[1] for item in sorted_matches]
                return categories, confidences
        
        return None

    def _partialMatch(self, text: str) -> Tuple[List[str], List[float]]:
        """
            Performs classification using partial substring matching as a final, low-confidence resort.

            Args:
                text (str): The cleaned input string to be classified.

            Returns:
                Tuple[List[str], List[float]]: A tuple containing a list of matching standardized 
                            category names and a corresponding list of low confidence scores. 
                            Returns `None` if no partial match is found.
        """
        category_scores = {}
        
        for main_category in FURNITURE_TYPE.keys():
            main_cat_lower = main_category.lower()
            
            # Check if main category is substring of text or vice versa
            if (main_cat_lower in text and len(main_cat_lower) >= 4) or \
               (text in main_cat_lower and len(text) >= 4):
                confidence = 0.75  # Lower confidence for partial matches
                category_scores[main_category] = confidence
            
            # Check synonyms too
            for synonym in FURNITURE_TYPE[main_category]:
                syn_lower = synonym.lower()
                if (syn_lower in text and len(syn_lower) >= 4) or \
                   (text in syn_lower and len(text) >= 4):
                    confidence = 0.7
                    if main_category not in category_scores or category_scores[main_category] < confidence:
                        category_scores[main_category] = confidence

        if category_scores:
            sorted_matches = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
            categories = [item[0] for item in sorted_matches]
            confidences = [item[1] for item in sorted_matches]
            return categories, confidences
        
        return None
    
    @lru_cache(maxsize=2000)
    def classifyProductType(self, text: str) -> Tuple[List[str], List[float]]:
        """
            Main classification method that determines the product type from a text string 
            using multiple fallback strategies, ordered by decreasing confidence.

            The `@lru_cache` decorator ensures results for the same input text are 
            cached, improving performance for repeated queries.

            Args:
                text (str): The raw input string (e.g., a search query) to classify.

            Returns:
                Tuple[List[str], List[float]]: A tuple where the first element is a list of 
                            detected product types (e.g., ['Sofa', 'Chair']) and the second 
                            is a corresponding list of confidence scores (e.g., [0.95, 0.88]). 
                            Returns `["Unknown"], [0.0]` if no match is found.
        """
        if not text or not text.strip():
            return ["Unknown"], [0.0]

        text_clean = text.lower().strip()
        
        # Strategy 1: Exact regex matching (highest confidence)
        exact_matches = self._exactMatch(text_clean)
        if exact_matches:
            return exact_matches

        # Strategy 2: Fuzzy string matching
        fuzzy_matches = self._fuzzyMatch(text_clean)
        if fuzzy_matches:
            return fuzzy_matches

        # Strategy 3: Vector similarity matching
        vector_matches = self._vectorMatch(text_clean)
        if vector_matches:
            return vector_matches

        # Strategy 4: Partial word matching
        partial_matches = self._partialMatch(text_clean)
        if partial_matches:
            return partial_matches

        return ["Unknown"], [0.0]
