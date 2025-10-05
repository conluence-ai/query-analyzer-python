# Import necessary libraries
import json
import os

# Load furniture data mappings from JSON file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Database - Table and column Names
BRAND_TABLE = '"Brand"'
PRODUCT_TABLE = 'product'
COLUMN_NAME = 'name'

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

# Product extraction constants

# Priority rules for disambiguation
DISAMBIGUATION_RULES = {
    "chair": ["Armchair", "Lounge Chair", "Chaise Lounge"],  # Prefer Armchair for generic "chair"
    "recliner": ["Armchair", "Lounge Chair"],  # Could be either
    "deck chair": ["Chaise Lounge", "Lounge Chair"],  # Prefer Chaise Lounge
    "easy chair": ["Armchair", "Lounge Chair"],  # Could be either
    "divan": ["Sofa", "Chaise Lounge"],  # Could be either
    "fainting couch": ["Sofa", "Chaise Lounge"]  # Could be either
}

# Feature extratction constants

# Category mappings for better organization
CATEGORY_MAPPINGS = {
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

# Define contextual patterns that might not be exact matches
FEATURE_CONTEXTUAL_PATTERNS = [
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
FUZZY_PATTERNS = [
    # Common misspellings for materials
    (r'\b(lether|leater|leathr)\b', 'Leather'),
    (r'\b(fabrik|febric|fabic)\b', 'Fabric'),
    (r'\b(mettal|metel|matel)\b', 'Metal'),
            
    # Common misspellings for furniture parts
    (r'\b(cussion|cushon|cushin)\b', 'cushion'),
    (r'\b(armrest|armrst|armest)\b', 'armrest'),
]


# Price extraction constants

# Currency patterns for price extraction
CURRENCY_PATTERNS = {
    'INR': [r'₹', r'rs\.?', r'rupees?', r'inr'],
    'USD': [r'\$', r'dollars?', r'usd', r'dollar?'],
    'EUR': [r'€', r'euros?', r'eur', r'euro?'],
    'GBP': [r'£', r'pounds?', r'gbp', r'pounds?']
}

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

# Number word mappings
NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "single": "1", "double": "2", "triple": "3", "quad": "4"
}

# Classication extraction constants

# Seater patterns
SEATER_PATTERNS = [
    (r'\b(\d+)[\s-]?seater?\b', lambda m: f"{m.group(1)} Seater"),
    (r'\b(\d+)[\s-]?seat\b', lambda m: f"{m.group(1)} Seater"),
    (r'\b(\d+)[\s-]?person\b', lambda m: f"{m.group(1)} Seater"),
    (r'\b(\d+)[\s-]?people\b', lambda m: f"{m.group(1)} Seater"),
    (r'\bfor[\s-]?(\d+)\b', lambda m: f"{m.group(1)} Seater"),
]

# Style extraction constants

# Define contextual patterns for better style detection
STYLE_CONTEXTUAL_PATTERNS = [
    # Modern variations
    (r'\b(modern|contemporary|current|latest|new)\s+(style|design|look)\b', 'Modern'),
    (r'\b(sleek|minimal|clean)\s+(design|style|look)\b', 'Minimalistic'),
            
    # Traditional variations  
    (r'\b(traditional|classic|formal|elegant)\s+(style|design|look)\b', 'Classical'),
    (r'\b(vintage|retro|antique)\s+(style|design|look)\b', 'Classical'),
            
    # Rustic variations
    (r'\b(rustic|country|farmhouse|cottage)\s+(style|design|look)\b', 'Rustic'),
    (r'\b(wooden|wood|natural)\s+(rustic|country|farmhouse)\b', 'Rustic'),
            
    # Industrial variations
    (r'\b(industrial|factory|warehouse|urban)\s+(style|design|look)\b', 'Industrial'),
    (r'\b(metal|steel|iron)\s+(industrial|modern)\b', 'Industrial'),
            
    # Scandinavian variations
    (r'\b(scandinavian|nordic|danish|swedish|finnish)\s+(style|design|look)\b', 'Scandinavian'),
    (r'\b(scandi|hygge)\b', 'Scandinavian'),
            
    # Bohemian variations
    (r'\b(bohemian|boho|eclectic|artsy)\s+(style|design|look)\b', 'Bohemian'),
    (r'\b(colorful|artistic|free[\s-]?spirited)\b', 'Bohemian'),
]
