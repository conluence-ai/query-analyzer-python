# Import necessary libraries
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

@dataclass
class PriceRange:
    """ Price range model """
    min: Optional[float]
    max: Optional[float]
    currency: str = 'EUR'
    confidence: float = 0.0

@dataclass
class ParserResult:
    """ Represents the structured, extracted, and classified data derived from user input. """
    product_type: List[str]
    brand_name: Optional[str]
    product_name: Optional[str]
    features: List[str]
    styles: List[str]
    price_range: Optional[PriceRange]
    location: Optional[str]
    classification_summary: Optional[Dict[str, Any]]
    extras: Optional[List[str]]
    confidence_score: float
    original_query: Optional[str]
    suggested_query: Optional[str]