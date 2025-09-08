# Import necessary libraries
import re
import numpy as np
from collections import defaultdict
from logs.logger import loggerSetup, logger
from processor.model import AdvancedNERModel
from typing import Dict, List, Tuple, Optional, Set
from datasets.data import PriceRange, ParserResult
from processor.price_extractor import PriceExtractor
from processor.dataset_processor import DatasetProcessor

from config.constants import ( 
    COLORS,
    MATERIALS,
    PRODUCT_DICTIONARY,
    FURNITURE_CONFIG,
    ENTITY_TEMPLATES,
    FEATURE_DICTIONARY,
    LOCATION_PATTERNS
)

# Configure logging
loggerSetup()

class FurnitureParser:
    """Furniture parser with ML/NLP models"""
    
    def __init__(self, data_files: List[str] = None):
        """
            Initialize the furniture parser
            
            Args:
                data_files (List[str]): List of Flipkart CSV files for training
        """
        self.dataset_processor = DatasetProcessor(data_files or [])
        self.ner_model = AdvancedNERModel()
        self.price_extractor = PriceExtractor()
        
        # Enhanced knowledge base
        self.knowledge_base = FURNITURE_CONFIG
        
        # Model predictions storage
        self.last_predictions = {}

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

        # Build mappings from your FEATURE_DICTIONARY
        self._build_feature_mappings()
        
        logger.info("Enhanced ML Furniture Parser initialized")
    
    def train_models(self, use_dataset: bool = True):
        """
            Train ML models with available data
            
            Args:
                use_dataset (bool): Whether to use datasets for training
        """
        print("Training models with dataset:", use_dataset)
        training_data = []
        
        if use_dataset and self.dataset_processor.data_files:
            logger.info("Generating training data from datasets...")
            flipkart_training_data = self.dataset_processor.generateTrainingData()
            training_data.extend(flipkart_training_data)
        
        # Add synthetic training data
        synthetic_data = self._generate_synthetic_training_data()
        training_data.extend(synthetic_data)
        
        if training_data:
            logger.info(f"Training NER model with {len(training_data)} examples")
            self.ner_model.train_custom_ner(training_data, n_iter=50)
        else:
            logger.warning("No training data available")

    def _build_feature_mappings(self):
        """Build reverse mapping from synonyms to main features"""
        for main_feature, synonyms in FEATURE_DICTIONARY.items():
            # Map the main feature to itself
            self.synonym_to_feature[main_feature.lower()] = main_feature
            
            # Map all synonyms to the main feature
            for synonym in synonyms:
                self.synonym_to_feature[synonym.lower()] = main_feature
            
            # Assign category
            category = self.category_mappings.get(main_feature, 'Other')
            self.feature_to_category[main_feature] = category
    
    def _generate_synthetic_training_data(self) -> List[Tuple]:
        """
            Generate synthetic training data for furniture domain
            
            Returns:
                List[Tuple]: List of tuples with text and entities for training
        """
        synthetic_data = []
        
        for template, base_entities in ENTITY_TEMPLATES[:3]:  # Limit synthetic data
            for i in range(5):  # Generate 5 examples per template
                if '{product}' in template:

                    # Randomly choose a product type
                    furniture_types = [syn for syns in PRODUCT_DICTIONARY.values() for syn in syns]
                    [furniture_types.append(main) for main in PRODUCT_DICTIONARY.keys()]


                    product = np.random.choice(furniture_types)
                    text = template.replace('{product}', product)
                    entities = [(start, end, label) for start, end, label in base_entities]
                    
                    # Adjust entity positions (simplified)
                    synthetic_data.append((text, {'entities': entities}))
        
        return synthetic_data
    
    def parse(self, query: str) -> ParserResult:
        """
            Enhanced parsing with ML models

            Args:
                query (str): Input text to parse

            Returns:
                ParserResult: Parsed result with product type, features, price range, location, and confidence    
        """
        logger.info(f"ML parsing query: {query}")
        
        # Reset predictions
        self.last_predictions = {}
        
        # Extract entities using multiple methods
        spacy_entities = self.ner_model.extract_entities_spacy(query)
        bert_entities = self.ner_model.extract_entities_bert(query)
        
        # Classify product type using zero-shot
        product_type, product_confidence = self.ner_model.classifyProductTypeZeroShot(query)
        
        # Store model predictions
        self.last_predictions = {
            'spacy_entities': spacy_entities,
            'bert_entities': bert_entities,
            'product_type_zero_shot': (product_type, product_confidence)
        }
        
        # Combine entity results
        combined_entities = self._combine_entity_results(spacy_entities, bert_entities, query)
        
        # Extract features using hybrid approach
        features = self._extractFeaturesML(query, combined_entities)

        # Extract 'other' items
        other_items = self._extract_other_items(query, combined_entities)
        
        # Extract price using hybrid extractor
        price_range = self.price_extractor.extractPriceRange(query)
        
        # Extract location
        location = self._extract_location_ml(query, combined_entities)
        
        # Calculate overall confidence
        confidence = self._calculate_ml_confidence(
            combined_entities, features, price_range, location, product_confidence
        )
        
        result = ParserResult(
            product_type=[product_type] if product_type != "Unknown" else [],
            features=features,
            other_features=other_items,
            price_range=price_range,
            location=location,
            confidence=confidence,
            # raw_entities=combined_entities,
            # model_predictions=self.last_predictions
        )
        
        logger.info(f"ML parsing completed with confidence: {confidence:.2f}")
        return result
    
    def _combine_entity_results(self, spacy_entities: Dict, bert_entities: Dict, query: str) -> Dict[str, List[str]]:
        """
            Combine entity results from different models
            
            Args:
                spacy_entities (Dict): Entities extracted by spaCy model
                bert_entities (Dict): Entities extracted by BERT model
                query (str): Input query text
                
            Returns:
                Dict[str, List[str]]: Combined entities with labels as keys and lists of entity texts as values
        """
        combined = defaultdict(list)
        
        # Add spaCy results
        for entity_type, entities in spacy_entities.items():
            combined[entity_type.lower()].extend(entities)
        
        # Add BERT results
        for entity_type, entities in bert_entities.items():
            combined[entity_type.lower()].extend(entities)
        
        # Add rule-based fallback for missing entities
        fallback_entities = self._extract_entities_rule_based(query)
        for entity_type, entities in fallback_entities.items():
            combined[entity_type].extend(entities)
        
        # Remove duplicates and clean
        for entity_type in combined:
            combined[entity_type] = list(set(combined[entity_type]))
            combined[entity_type] = [e.strip().lower() for e in combined[entity_type] if e.strip()]
        
        return dict(combined)
    
    def _extract_entities_rule_based(self, text: str) -> Dict[str, List[str]]:
        """
            Rule-based entity extraction as fallback
            
            Args:
                text (str): Input text to extract entities from
                
            Returns:
                Dict[str, List[str]]: Dictionary of entities with labels as keys and lists of entity texts as values
        """
        entities = defaultdict(list)
        text_lower = text.lower()
        
        # Product extraction
        for product_type, variants in self.knowledge_base['products'].items():
            for variant in variants:
                if variant in text_lower:
                    entities['product'].append(variant)
        
        # Material extraction
        for material in MATERIALS:
            if material in text_lower:
                entities['material'].append(material)
        
        # Color extraction
        for color in COLORS:
            if color in text_lower:
                entities['color'].append(color)
        
        # Location extraction
        for location in self.knowledge_base['locations']:
            if location in text_lower:
                entities['location'].append(location)
        
        return dict(entities)
    
    def _extractFeaturesML(self, text: str, entities: Dict[str, List[str]]) -> List[str]:
        """
            Extract features using ML-enhanced approach

            Args:
                text (str): Input text to analyze
                entities (Dict[str, List[str]]): Extracted entities from text

            Returns:
                Dict[str, List[str]]: Extracted features with categories as keys and lists of feature names as values    
        """
        features = self._extract_features_ml_improved(text, entities)
        
        # # Post-process to resolve conflicts and add inferences
        # features = self._post_process_features(features)
        
        return features
    
    def _extract_features_ml_improved(self, text: str, entities: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Improved feature extraction with better accuracy
        """
        features = defaultdict(set)  # Use sets to avoid duplicates
        text_lower = text.lower()
        
        # Method 1: Direct keyword matching
        detected_features = self._extract_by_keywords(text_lower)
        for feature in detected_features:
            category = self.feature_to_category.get(feature, 'Other')
            features[category].add(feature)
        
        # Method 2: Contextual phrase matching
        contextual_features = self._extract_by_context(text_lower)
        for feature in contextual_features:
            category = self.feature_to_category.get(feature, 'Other')
            features[category].add(feature)
        
        # # # Method 3: Entity-based feature mapping
        # # entity_features = self._extract_from_entities(entities)
        # # for feature in entity_features:
        # #     category = self.feature_to_category.get(feature, 'Other')
        # #     features[category].add(feature)
        
        # # # Method 4: Pattern-based extraction for complex features
        # # pattern_features = self._extract_by_patterns(text_lower)
        # # for feature in pattern_features:
        # #     category = self.feature_to_category.get(feature, 'Other')
        # #     features[category].add(feature)
        
        # # Method 5: Contextual material-part associations
        # self._associate_materials_with_parts_improved(text_lower, entities, features)
        
        flat_features = []
        for category_features in features.values():
            flat_features.extend(category_features)
        
        return sorted(list(set(flat_features)))
        # # Convert sets back to lists and clean up
        # return {k: list(v) for k, v in features.items() if v}
    
    def _extract_by_patterns(self, text: str) -> Set[str]:
        """Extract features using advanced pattern matching"""
        detected = set()
        
        # Advanced patterns for complex features
        advanced_patterns = [
            # Piping patterns
            (r'\b(piping|welting|trim|edging)\b', 'Flat Piping'),
            (r'\b(leather\s+piping|leather\s+trim)\b', 'Leather Piping'),
            (r'\b(braid|braided)\s+(piping|trim)\b', 'Braid Piping'),
            
            # Upholstery patterns
            (r'\b(upholstered|padded|cushioned)\s+(base|plinth)\b', 'Upholstered Plinth'),
            (r'\b(upholstered|padded)\s+(shell|structure)\b', 'Upholstered Shell'),
            
            # Tufting patterns
            (r'\b(grid|diamond|square)\s+(tufted|quilted)\b', 'Grid Tufted Back'),
            (r'\b(horizontal|lateral)\s+(tufting|quilting)\b', 'Horizontal Tufting'),
            (r'\b(quilted|tufted)\s+(back|backrest)\b', 'Quilted Back'),
            
            # Construction patterns
            (r'\b(solid|heavy|substantial)\s+(base|foundation)\b', 'Heavy Base'),
            (r'\b(modular|detachable|removable)\s+(arm|arms)\b', 'Modular Arm'),
            (r'\b(integrated|built[\s-]?in)\s+(arm|arms)\b', 'Armrest integrated with Structure'),
        ]
        
        for pattern, feature in advanced_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                detected.add(feature)
        
        return detected
    
    def _associate_materials_with_parts_improved(self, text: str, entities: Dict, features: defaultdict):
        """
        Improved material-part association with better context awareness
        """
        words = text.split()
        
        # Define proximity patterns
        material_part_patterns = [
            # Leg associations
            (r'\b(metal|steel|chrome|iron)\s+(?:\w+\s+){0,2}(leg|legs)\b', 'Metal Legs'),
            (r'\b(wooden|wood|timber)\s+(?:\w+\s+){0,2}(leg|legs)\b', 'Wooden Legs'),
            
            # Base associations
            (r'\b(metal|steel)\s+(?:\w+\s+){0,2}(base|plinth|platform)\b', 'Metal plinth'),
            (r'\b(wooden|wood)\s+(?:\w+\s+){0,2}(base|plinth|platform)\b', 'Wooden plinth'),
            (r'\b(upholstered|padded|cushioned)\s+(?:\w+\s+){0,2}(base|plinth)\b', 'upholstered base'),
            
            # Arm associations
            (r'\b(metal)\s+(?:\w+\s+){0,2}(arm|arms)\b', 'armsWithMetalDetail'),
            (r'\b(padded|cushioned)\s+(?:\w+\s+){0,2}(arm|arms)\b', 'With Armrest'),
            
            # Structure associations
            (r'\b(metal|steel)\s+(?:\w+\s+){0,2}(frame|structure)\b', 'Metal Structure'),
            (r'\b(wire|wireframe)\s+(?:\w+\s+){0,2}(frame|structure)\b', 'Metal Wire Frame'),
        ]
        
        for pattern, feature in material_part_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                category = self.feature_to_category.get(feature, 'Other')
                features[category].add(feature)
    
    def _extract_from_entities(self, entities: Dict[str, List[str]]) -> Set[str]:
        """Extract features from named entities"""
        detected = set()
        
        # Map entity types to features
        entity_mappings = {
            'material': {
                'leather': 'Leather Back Covering',
                'fabric': 'Fabric',
                'metal': 'Metal detail',
                'wood': 'Wooden plinth',
                'wooden': 'Wooden plinth'
            },
            'color': {
                # Colors don't directly map to structural features in your dictionary
                # but could be used for material context
            },
            'style': {
                'chesterfield': 'Chesterfield',
                'tufted': 'Quilted Back',
                'sectional': 'LShape'
            }
        }
        
        for entity_type, entity_list in entities.items():
            if entity_type in entity_mappings:
                for entity in entity_list:
                    entity_lower = entity.lower()
                    if entity_lower in entity_mappings[entity_type]:
                        feature = entity_mappings[entity_type][entity_lower]
                        detected.add(feature)
        
        return detected
    
    def _extract_by_context(self, text: str) -> Set[str]:
        """Extract features using contextual phrase matching"""
        detected = set()
        
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
        
        for pattern, feature in contextual_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                detected.add(feature)
        
        return detected
    
    def _is_contextually_relevant(self, phrase: str, words: List[str], position: int) -> bool:
        """Check if the detected phrase is contextually relevant"""
        # Add logic to prevent false positives
        # For example, only allow "leather" if it's near furniture terms
        
        if phrase == "leather":
            # Check if leather is mentioned with furniture parts
            context_window = words[max(0, position-2):min(len(words), position+3)]
            furniture_parts = ['sofa', 'chair', 'back', 'seat', 'arm', 'cushion']
            return any(part in ' '.join(context_window) for part in furniture_parts)
        
        return True 
    
    def _extract_other_items(self, text: str, entities: Dict[str, List[str]]) -> List[str]:
        """Extract items that should go in 'other' category"""
        other_items = []
        text_lower = text.lower()
        
        # Furniture type modifiers that go in "other"
        modifiers = [
            'reclining', 'recliner', 'sectional', 'sleeper', 'convertible',
            'modular', 'corner', 'chaise', 'ottoman', 'bench'
        ]
        
        for modifier in modifiers:
            if modifier in text_lower:
                # Add the modifier + main product type
                for product_type in self.knowledge_base['products'].keys():
                    if any(variant in text_lower for variant in self.knowledge_base['products'][product_type]):
                        other_items.append(f"{modifier} {product_type}")
                        break
        
        return other_items
    
    def _post_process_features(self, features: List[str]) -> List[str]:
        """
        Post-process features to resolve conflicts and add inferences
        """
        processed = dict(features)
    
        # ----- Conflict resolution -----
        if "Arms" in processed:
            arms_features = processed["Arms"]
            if "With Armrest" in arms_features and "Without Armrest" in arms_features:
                arms_features.remove("With Armrest")  # keep "Without Armrest"
        
        if "Legs" in processed:
            legs_features = processed["Legs"]
            if "Wooden Legs" in legs_features and "Metal Legs" in legs_features:
                # Keep both OR decide a priority
                pass  
    
        # ----- Add inferences -----
        if "Style" in processed and "Chesterfield" in processed["Style"]:
            processed.setdefault("Back", []).append("Quilted Back")
        
        flat_features = []
        for values in processed.values():
            flat_features.extend(values)

        # Deduplicate and clean
        return sorted(set(flat_features))
    
    def _extract_by_keywords(self, text: str) -> Set[str]:
        """Extract features by direct keyword matching"""
        detected = set()
        words = text.split()
        
        # Only check for exact matches, not broad synonyms
        for window_size in [1, 2, 3]:  # Reduced window size
            for i in range(len(words) - window_size + 1):
                phrase = ' '.join(words[i:i + window_size])
                
                # Only add if exact match in synonym dictionary
                if phrase in self.synonym_to_feature:
                    main_feature = self.synonym_to_feature[phrase]
                    # Additional filtering to prevent over-generation
                    if self._is_contextually_relevant(phrase, words, i):
                        detected.add(main_feature)
        
        return detected
    
    def _associate_materials_with_parts(self, text: str, entities: Dict, features: Dict):
        """
            Associate materials with furniture parts based on context
            
            Args:
                text (str): Input text to analyze
                entities (Dict): Extracted entities from text
                features (Dict): Extracted features to update
        """
        # Look for material-part associations in proximity
        words = text.split()
        
        for i, word in enumerate(words):
            if word in entities.get('material', []):
                # Check nearby words for parts
                context_window = words[max(0, i-3):min(len(words), i+4)]
                
                if any(leg_word in context_window for leg_word in ['leg', 'legs']):
                    if word == 'metal':
                        features['Legs'].append('Metal Leg')
                        features['Structure'].append('Metal Detail')
                    elif word == 'wood' or word == 'wooden':
                        features['Legs'].append('Wooden Leg')
                
                if any(arm_word in context_window for arm_word in ['arm', 'arms']):
                    if word == 'metal':
                        features['Arms'].append('Metal Arms')
                    elif word == 'padded':
                        features['Arms'].append('Padded Arms')
    
    def _extract_location_ml(self, text: str, entities: Dict[str, List[str]]) -> Optional[str]:
        """
            Extract location using ML approach

            Args:
                text (str): Input text to analyze
                entities (Dict[str, List[str]]): Extracted entities from text

            Returns:
                Optional[str]: Extracted location or None if not found    
        """
        locations = entities.get('location', [])
        
        if locations:
            # Format and return the most confident location
            location = locations[0]
            return location.title().replace('_', ' ')
        
        for pattern in LOCATION_PATTERNS:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                location_candidate = match.strip()
                if location_candidate in self.knowledge_base['locations']:
                    return location_candidate.title()
        
        return None
    
    def _calculate_ml_confidence(self, entities: Dict, features: Dict, 
                                price_range: Optional[PriceRange], 
                                location: Optional[str], 
                                product_confidence: List[float]) -> float:
        """
            Calculate confidence score using ML insights
            
            Args:
                entities (Dict): Extracted entities from text
                features (Dict): Extracted features from text
                price_range (Optional[PriceRange]): Extracted price range
                location (Optional[str]): Extracted location
                product_confidence (float): Confidence score from product type classification

            Returns:
                float: Overall confidence score (0.0 to 1.0)
        """
        score = 0.0
        
        # Product type confidence from zero-shot classification
        if isinstance(product_confidence, (list, tuple)):
            score += sum(product_confidence) * 0.3
        else:
            score += product_confidence * 0.3
        
        # Entity extraction confidence
        entity_count = sum(len(v) for v in entities.values())
        entity_score = min(entity_count / 6, 1.0) * 0.25
        score += entity_score
        
        # Feature extraction confidence
        if features:
            feature_diversity = len(features)
            feature_score = min(feature_diversity / 4, 1.0) * 0.2
            score += feature_score
        
        # Price extraction confidence
        if price_range:
            price_confidence = getattr(price_range, 'confidence', 0.5)
            score += price_confidence * 0.15
        
        # Location extraction confidence
        if location:
            score += 0.1
        
        return min(score, 1.0)
    
    def inputToDict(self, query: str) -> Dict:
        """
            Parse query and return as dictionary

            Args:
                query (str): Input text to parse

            Returns:
                Dict: Parsed result with product type, features, price range, location, and confidence    
        """
        result = self.parse(query)
        
        output = {
            "product_type": result.product_type,
            "features": result.features,
            "other_features": result.other_features, 
        }
        
        if result.price_range:
            output["price_range"] = {
                "min": result.price_range.min,
                "max": result.price_range.max,
                "currency": result.price_range.currency,
                "confidence": getattr(result.price_range, 'confidence', 0.0)
            }
        
        if result.location:
            output["location"] = result.location
        
        # Add ML model insights
        output["confidence"] = result.confidence
        # output["ml_insights"] = {
        #     "spacy_entities": result.model_predictions.get('spacy_entities', {}),
        #     "bert_entities": result.model_predictions.get('bert_entities', {}),
        #     "zero_shot_prediction": result.model_predictions.get('product_type_zero_shot', ("Unknown", 0.0))
        # }
        
        return output
    
    def evaluate_on_test_data(self, test_queries: List[Tuple[str, Dict]]) -> Dict:
        """
            Evaluate the parser on test data
            
            Args:
                test_queries (List[Tuple[str, Dict]]): List of tuples with query and expected result

            Returns:
                Dict: Evaluation results with accuracy, precision, recall, F1 score, and detailed results
        """
        results = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'detailed_results': []
        }
        
        correct_predictions = 0
        total_predictions = len(test_queries)
        
        for query, expected_result in test_queries:
            predicted_result = self.inputToDict(query)
            
            # Simple accuracy calculation (can be enhanced)
            is_correct = (
                predicted_result.get('product_type') == expected_result.get('product_type') and
                len(predicted_result.get('features', {})) > 0
            )
            
            if is_correct:
                correct_predictions += 1
            
            results['detailed_results'].append({
                'query': query,
                'expected': expected_result,
                'predicted': predicted_result,
                'correct': is_correct
            })
        
        results['accuracy'] = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return results