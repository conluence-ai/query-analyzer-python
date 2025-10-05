"""Furniture Parser using ML/NLP models and hybrid techniques"""

# Import necessary libraries
import re
import logging

# Import custom modules
from config.config import ParserResult
from query_parser.product_type_extractor import ProductTypeExtractor
from query_parser.feature_extractor import FeatureExtractor
from query_parser.price_extractor import PriceExtractor
from query_parser.style_extractor import StyleExtractor
from query_parser.classification_extractor import StyleClassification
from query_parser.brand_product_extractor import BrandProductExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

class FurnitureParser:
    """ Furniture parser with ML/NLP models """
    
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
        self.feature_extractor = FeatureExtractor()
    
    def structureQuery(self, query: str) -> ParserResult:
        """
            Enhanced parsing with ML models
            
            Args:
                query (str): Input text to parse

            Returns:
                ParserResult: Parsed result with product type, features, price range, location, and confidence    
        """
        logger.info(f"ML parsing query: {query}")
        
        # STEP 1: Extract brand/product FIRST (before anything else)
        data = self.brand_product_extractor.extractProductBrand(query)
        
        # Safely extract brand name
        brand_name = data['brand']['name'] if data and data.get('brand') else None
        logger.info(f"Extracted brand: {brand_name}")
        
        # Verify and extract product name
        product_name = None
        if data and data.get('product'):
            extracted_product = data['product']['name']
            if self._verifyProductInQuery(extracted_product, query):
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
        features = self.feature_extractor.extractFeatures(query)
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
    
    def _verifyProductInQuery(self, product_name: str, query: str) -> bool:
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