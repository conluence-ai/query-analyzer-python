# Import necessary libraries
import re
from collections import Counter
from datasets.data import PriceRange
from typing import List, Tuple, Optional
from config.constants import CURRENCY_PATTERNS, PRICE_PATTERNS, NUMBER_PATTERNS, CONTEXT_WORDS

class PriceExtractor:
    """Advanced price extraction using multiple methods"""
    
    def __init__(self):
        """
            Initialize the price extractor

            This sets up the currency patterns, price patterns, and number patterns
            for extracting prices from text.
        """
        self.currency_patterns = CURRENCY_PATTERNS
        
        # Enhanced price patterns with context
        self.price_patterns = PRICE_PATTERNS
        
        # Duckling-like number extraction patterns
        self.number_patterns = NUMBER_PATTERNS
    
    def extractPriceRange(self, text: str) -> Optional[PriceRange]:
        """
            Extract price range using hybrid approach
            
            Args:
                text (str): Input text to analyze
            
            Returns:
                Optional[PriceRange]: Extracted price range with currency and confidence
        """
        text_lower = text.lower()
        
        # Detect currency
        currency = self._detectCurrency(text_lower)
        
        # Extract price values using multiple methods
        price_results = []
        
        # Pattern matching
        pattern_result = self._extractUsingPatterns(text_lower, currency)
        if pattern_result:
            price_results.append(('pattern', pattern_result, 0.8))
        
        # Number extraction + context
        context_result = self._extract_using_context(text_lower, currency)
        if context_result:
            price_results.append(('context', context_result, 0.6))
        
        # ML-based extraction (simplified)
        ml_result = self._extract_using_ml(text_lower, currency)
        if ml_result:
            price_results.append(('ml', ml_result, 0.7))
        
        # Combine results using weighted voting
        if price_results:
            return self._combine_price_results(price_results)
        
        return None
    
    def _detectCurrency(self, text: str) -> str:
        """
            Detect currency from text

            Args:
                text (str): Input text to analyze

            Returns:
                str: Detected currency code (INR, USD, EUR, GBP)    
        """
        currency_scores = {}
        
        for currency, patterns in self.currency_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                score += matches
            currency_scores[currency] = score
        
        # Default to INR if no currency detected or if rupees mentioned
        if not currency_scores or max(currency_scores.values()) == 0:
            return 'INR'
        
        return max(currency_scores, key=currency_scores.get)
    
    def _extractUsingPatterns(self, text: str, currency: str) -> Optional[PriceRange]:
        """Extract price using regex patterns"""
        for pattern in self.price_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if isinstance(matches[0], tuple) and len(matches[0]) == 2:
                    # Range pattern
                    min_val = self._parse_number(matches[0][0])
                    max_val = self._parse_number(matches[0][1])
                    if min_val and max_val:
                        return PriceRange(min=min_val, max=max_val, currency=currency, confidence=0.8)
                else:
                    # Single value pattern
                    price_val = self._parse_number(matches[0])
                    if price_val:
                        # Determine if it's min or max based on context
                        if any(word in text for word in CONTEXT_WORDS['max']):
                            return PriceRange(min=None, max=price_val, currency=currency, confidence=0.8)
                        elif any(word in text for word in CONTEXT_WORDS['min']):
                            return PriceRange(min=price_val, max=None, currency=currency, confidence=0.8)
                        elif any(word in text for word in CONTEXT_WORDS['exact']):
                            return PriceRange(min=price_val, max=price_val, currency=currency, confidence=0.9)
                        else:
                            return PriceRange(min=None, max=price_val, currency=currency, confidence=0.7)
        
        return None
    
    def _extract_using_context(self, text: str, currency: str) -> Optional[PriceRange]:
        """Extract price using context analysis"""
        # Find all numbers in text
        numbers = []
        for pattern in self.number_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                parsed_num = self._parse_number(match)
                if parsed_num and 100 <= parsed_num <= 10000000:  # Reasonable furniture price range
                    numbers.append(parsed_num)
        
        if not numbers:
            return None
        
        price_type = 'max'  # Default
        for p_type, words in CONTEXT_WORDS.items():
            if any(word in text for word in words):
                price_type = p_type
                break
        
        price_val = max(numbers) if len(numbers) > 1 else numbers[0]
        
        if price_type == 'max':
            return PriceRange(min=None, max=price_val, currency=currency, confidence=0.6)
        elif price_type == 'min':
            return PriceRange(min=price_val, max=None, currency=currency, confidence=0.6)
        else:
            return PriceRange(min=None, max=price_val, currency=currency, confidence=0.5)
    
    def _extract_using_ml(self, text: str, currency: str) -> Optional[PriceRange]:
        """
            Extract price using ML approach (simplified)
            
            Args:
                text (str): Input text to analyze
                currency (str): Detected currency code

            Returns:
                Optional[PriceRange]: Extracted price range with currency and confidence
        """
        # This is a placeholder for ML-based price extraction
        # In practice, you would train a model on labeled data
        
        # Simple heuristic: look for largest reasonable number
        numbers = re.findall(r'\d+(?:,\d+)*(?:\.\d+)?', text)
        if numbers:
            parsed_numbers = [self._parse_number(num) for num in numbers]
            valid_numbers = [n for n in parsed_numbers if n and 100 <= n <= 10000000]
            
            if valid_numbers:
                max_price = max(valid_numbers)
                return PriceRange(min=None, max=max_price, currency=currency, confidence=0.4)
        
        return None
    
    def _parse_number(self, num_str: str) -> Optional[float]:
        """
            Parse number string to float
            
            Args:
                num_str (str): Number string to parse

            Returns:
                Optional[float]: Parsed number as float, or None if invalid
        """
        if not num_str:
            return None
        
        try:
            # Handle special cases
            if 'k' in num_str.lower() or 'thousand' in num_str.lower():
                base_num = float(re.sub(r'[^\d.]', '', num_str))
                return base_num * 1000
            elif 'l' in num_str.lower() or 'lakh' in num_str.lower() or 'lac' in num_str.lower():
                base_num = float(re.sub(r'[^\d.]', '', num_str))
                return base_num * 100000
            elif 'cr' in num_str.lower() or 'crore' in num_str.lower():
                base_num = float(re.sub(r'[^\d.]', '', num_str))
                return base_num * 10000000
            else:
                # Regular number
                clean_num = re.sub(r'[,\s]', '', num_str)
                return float(clean_num)
        except ValueError:
            return None
    
    def _combine_price_results(self, results: List[Tuple]) -> PriceRange:
        """
            Combine multiple price extraction results

            Args:
                results (List[Tuple]): List of tuples with method, PriceRange, and confidence

            Returns:
                PriceRange: Combined price range with weighted average and confidence    
        """
        # Weighted average based on confidence
        min_prices = []
        max_prices = []
        currencies = []
        total_confidence = 0
        
        for method, price_range, confidence in results:
            if price_range.min:
                min_prices.append((price_range.min, confidence))
            if price_range.max:
                max_prices.append((price_range.max, confidence))
            currencies.append((price_range.currency, confidence))
            total_confidence += confidence
        
        # Calculate weighted averages
        final_min = None
        if min_prices:
            weighted_min = sum(price * conf for price, conf in min_prices) / sum(conf for _, conf in min_prices)
            final_min = weighted_min
        
        final_max = None
        if max_prices:
            weighted_max = sum(price * conf for price, conf in max_prices) / sum(conf for _, conf in max_prices)
            final_max = weighted_max
        
        # Most confident currency
        currency_counts = Counter(curr for curr, _ in currencies)
        final_currency = currency_counts.most_common(1)[0][0] if currencies else 'INR'

        if final_min is not None:
            final_min = round(final_min, 2)
        if final_max is not None:
            final_max = round(final_max, 2)
        
        return PriceRange(
            min=final_min,
            max=final_max,
            currency=final_currency,
            confidence=min(total_confidence / len(results), 1.0)
        )