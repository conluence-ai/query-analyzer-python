# Import necessary libraries
import json
import os

# Load furniture data mappings from JSON file

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load furniture categories from JSON file
with open(os.path.join(BASE_DIR, "dictionary", "furniture_features.json")) as f:
    FURNITURE_CATEGORY = json.load(f)

# Load product types from JSON file
with open(os.path.join(BASE_DIR, "dictionary", "furniture_type.json")) as f:
    FURNITURE_TYPE = json.load(f)

# Load furniture styles from JSON file
with open(os.path.join(BASE_DIR, "dictionary", "furniture_styles.json")) as f:
    FURNITURE_STYLES = json.load(f)

# Load furniture classification from JSON file
with open(os.path.join(BASE_DIR, "dictionary", "furniture_classification.json")) as f:
    FURNITURE_CLASSIFICATION = json.load(f)

# Currency patterns for price extraction
CURRENCY_PATTERNS = {
    'INR': [r'₹', r'rs\.?', r'rupees?', r'inr'],
    'USD': [r'\$', r'dollars?', r'usd', r'dollar?'],
    'EUR': [r'€', r'euros?', r'eur', r'euro?'],
    'GBP': [r'£', r'pounds?', r'gbp', r'pounds?']
}

# Price extraction patterns

# Updated Price extraction patterns that handle spaces before suffixes
PRICE_PATTERNS = [
    # Indian currency contextual patterns
    r'(?:under|below|less\s+than|within|up\s+to|max|maximum)\s*₹?\s*(\d+(?:,\d+)*(?:\.\d+)?\s*(?:k|K|thousand|l|L|lakh|lac|cr|crore)?)\s*(?:rupees?|rs\.?|₹)?',
    r'(?:above|over|more\s+than|starting\s+from|min|minimum)\s*₹?\s*(\d+(?:,\d+)*(?:\.\d+)?\s*(?:k|K|thousand|l|L|lakh|lac|cr|crore)?)\s*(?:rupees?|rs\.?|₹)?',

    # Explicit ranges
    r'₹\s*(\d+(?:,\d+)*(?:\.\d+)?\s*(?:k|K|thousand|l|L|lakh|lac|cr|crore)?)\s*(?:to|-)\s*₹?\s*(\d+(?:,\d+)*(?:\.\d+)?\s*(?:k|K|thousand|l|L|lakh|lac|cr|crore)?)',
    r'(\d+(?:,\d+)*(?:\.\d+)?\s*(?:k|K|thousand|l|L|lakh|lac|cr|crore)?)\s*(?:to|-)\s*(\d+(?:,\d+)*(?:\.\d+)?\s*(?:k|K|thousand|l|L|lakh|lac|cr|crore)?)(?:\s*(?:rupees?|rs\.?|₹))?',
    r'price\s*(?:range|is)?\s*₹?\s*(\d+(?:,\d+)*(?:\.\d+)?\s*(?:k|K|thousand|l|L|lakh|lac|cr|crore)?)\s*(?:to|-)\s*₹?\s*(\d+(?:,\d+)*(?:\.\d+)?\s*(?:k|K|thousand|l|L|lakh|lac|cr|crore)?)(?:\s*(?:rupees?|rs\.?|₹))?',

    # Between X and Y
    r'(?:between)\s*₹?\s*(\d+(?:,\d+)*(?:\.\d+)?\s*(?:k|K|thousand|l|L|lakh|lac|cr|crore)?)\s*(?:and|-)\s*₹?\s*(\d+(?:,\d+)*(?:\.\d+)?\s*(?:k|K|thousand|l|L|lakh|lac|cr|crore)?)(?:\s*(?:rupees?|rs\.?|₹|\$|dollars?|€|euros?|£|pounds?))?',

    # Budget style
    r'budget\s*(?:of|is|around)?\s*₹?\s*(\d+(?:,\d+)*(?:\.\d+)?\s*(?:k|K|thousand|l|L|lakh|lac|cr|crore)?)(?:\s*(?:rupees?|rs\.?|₹))?',

    # Single values (catch-all)
    r'(\d+(?:,\d+)*(?:\.\d+)?\s*(?:k|K|thousand|l|L|lakh|lac|cr|crore)?)(?:\s*(?:rupees?|rs\.?|₹|\$|dollars?|€|euros?|£|pounds?))',
    r'(?:₹|\$|€|£)\s*(\d+(?:,\d+)*(?:\.\d+)?\s*(?:k|K|thousand|l|L|lakh|lac|cr|crore)?)',
]

# Number patterns for price extraction
NUMBER_PATTERNS = [
    r'(\d+(?:\.\d+)?)\s*(?:k|K|thousand|thous)',  # 5k, 10K, 30k, 10 thousand
    r'(\d+(?:\.\d+)?)\s*(?:l|L|lakh|lac|Lakh|Lac)',  # 2 lakh, 5L
    r'(\d+(?:\.\d+)?)\s*(?:cr|CR|crore|Crore)',   # 1 crore, 2cr
    r'(\d+(?:,\d+)*(?:\.\d+)?)',  # Standard numbers with commas (keep this last)
]

# Pattern for price ranges: X-Y, X to Y, between X and Y
RANGE_PATTERNS = [
    r'(\d+[\d,.]*)\s*[-–—to]\s*(\d+[\d,.]*)',  # X-Y or X to Y
    r'between\s+(\d+[\d,.]*)\s+and\s+(\d+[\d,.]*)',  # between X and Y
    r'(\d+[\d,.]*)\s*and\s*(\d+[\d,.]*)'  # X and Y
]

# Analyze context around numbers
CONTEXT_WORDS = {
    'max': ['under', 'below', 'less', 'within', 'budget', 'maximum', 'max', 'up to'],
    'min': ['above', 'over', 'more', 'starting', 'minimum', 'min', 'from'],
    'range': ['between', 'to', 'and', 'from', 'range'],
    'exact': ['price', 'cost', 'worth', 'value', 'amount', 'total', 'around', 'about']
}

# Location patterns
LOCATION_PATTERNS = [
    r'in\s+([a-zA-Z\s]+)(?:\s|$)',
    r'at\s+([a-zA-Z\s]+)(?:\s|$)',
    r'from\s+([a-zA-Z\s]+)(?:\s|$)'
]