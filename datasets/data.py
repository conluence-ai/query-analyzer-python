# Import necessary libraries
import nltk
from typing import Optional, Dict, List, Any
from dataclasses import dataclass


# Download required NLTK data
def downloadNltkData():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

@dataclass
class PriceRange:
    min: Optional[float]
    max: Optional[float]
    currency: str
    confidence: float = 0.0

@dataclass
class ParserResult:
    product_type: List[str]
    features: List[str]
    other_features: List[str]
    price_range: Optional[PriceRange]
    location: Optional[str]
    confidence: float
    # raw_entities: Optional[Dict[str, List[str]]]
    # model_predictions: Optional[Dict[str, Any]]

# Future addition
# style,
#     brand_name,
#     product_name DB,
# classificationSummary - optional dict -> variantType - 1-seater, 2 seater, 3 seater compact
# extras