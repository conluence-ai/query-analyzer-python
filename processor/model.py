# Import necessary libraries
import re
import spacy
import torch
from logs.logger import loggerSetup, logger
from spacy.training import Example
from collections import defaultdict
from typing import Dict, List, Tuple
from spacy.util import minibatch, compounding
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification, pipeline,
)
from config.constants import ENTITY_LABELS, PRODUCT_DICTIONARY


# Configure logging
loggerSetup()

class AdvancedNERModel:
    """Advanced NER using spaCy and BERT"""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
            Initialize the NER model
            
            Args:
                model_name (str): Name of the spaCy model to load
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.warning(f"Model {model_name} not found, using blank model")
            self.nlp = spacy.blank("en")
        
        # Add custom NER component if not present
        if "ner" not in self.nlp.pipe_names:
            self.ner = self.nlp.add_pipe("ner")
        else:
            self.ner = self.nlp.get_pipe("ner")
        
        # BERT model for contextual understanding
        self.bert_tokenizer = None
        self.bert_model = None
        self._load_bert_model()
        
        # Zero-shot classifier for product types
        self.zero_shot_classifier = None
        self.confidence_threshold = 0.3
        self._load_zero_shot_classifier()
        
    def _load_bert_model(self):
        """
            Load BERT model for NER. This uses a pre-trained BERT model fine-tuned on NER tasks.    
        """
        try:
            model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
            self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert_model = AutoModelForTokenClassification.from_pretrained(model_name)
            logger.info("BERT NER model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load BERT model: {e}")
    
    def _load_zero_shot_classifier(self):
        """
            Load zero-shot classifier. This uses a pre-trained model for zero-shot classification tasks.    
        """
        try:
            self.zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
            logger.info("Zero-shot classifier loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load zero-shot classifier: {e}")

    def remove_overlaps(self, entities: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
        """
            Remove overlapping entity spans.

            Args:
                entities (List[Tuple[int, int, str]]): List of entity tuples (start, end, label)

            Returns:
                List[Tuple[int, int, str]]: Cleaned list of entities without overlaps    
        """
        entities = sorted(entities, key=lambda x: (x[0], -x[1]))
        clean = []
        prev_end = -1
        for start, end, label in entities:
            if start >= prev_end:
                clean.append((start, end, label))
                prev_end = end
        return clean
    
    def train_custom_ner(self, training_data: List[Tuple], n_iter: int = 100):
        """
            Train custom NER model with given dataset
            
            Args:
                training_data (List[Tuple]): List of tuples with text and annotations for training
                n_iter (int): Number of training iterations
        """
        logger.info(f"Training custom NER with {len(training_data)} examples")
        ents = []
        
        # Add entity labels
        for label in ENTITY_LABELS:
            self.ner.add_label(label)
        
        # Disable other pipes during training
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "ner"]
        with self.nlp.disable_pipes(*other_pipes):
            optimizer = self.nlp.begin_training()
            
            for iteration in range(n_iter):
                losses = {}
                # Batch the examples
                batches = minibatch(training_data, size=compounding(4.0, 32.0, 1.001))
                
                for batch in batches:
                    examples = []
                    for text, annotations in batch:
                        if "entities" in annotations:
                            annotations["entities"] = self.remove_overlaps(annotations["entities"])
                            
                        doc = self.nlp.make_doc(text)
                        example = Example.from_dict(doc, annotations)
                        examples.append(example)
                    
                    self.nlp.update(examples, drop=0.5, losses=losses)
                
                if iteration % 20 == 0:
                    logger.info(f"Iteration {iteration}, Losses: {losses}")
        
        logger.info("Custom NER training completed")
    
    def extract_entities_spacy(self, text: str) -> Dict[str, List[str]]:
        """
            Extract entities using trained spaCy model
            
            Args:
                text (str): Input text to extract entities from
                
            Returns:
                Dict[str, List[str]]: Dictionary of entities with labels as keys and lists of entity texts as values
        """
        doc = self.nlp(text)
        entities = defaultdict(list)
        
        for ent in doc.ents:
            entities[ent.label_].append(ent.text)
        
        return dict(entities)
    
    def extract_entities_bert(self, text: str) -> Dict[str, List[str]]:
        """
            Extract entities using BERT model
            
            Args:
                text (str): Input text to extract entities from
                
            Returns:
                Dict[str, List[str]]: Dictionary of entities with labels as keys and lists of entity texts as values
        """
        if not self.bert_model or not self.bert_tokenizer:
            return {}
        
        try:
            # Tokenize text
            inputs = self.bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=2)
            
            # Decode predictions (simplified)
            tokens = self.bert_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            entities = defaultdict(list)
            
            # This is a simplified entity extraction - in practice, you'd need proper BIO tagging
            current_entity = []
            current_label = None
            
            for token, pred in zip(tokens, predictions[0]):
                if token.startswith("##"):
                    continue
                if pred.item() > 0:  # Simplified: any non-O tag
                    current_entity.append(token)
                else:
                    if current_entity and current_label:
                        entity_text = " ".join(current_entity)
                        entities[current_label].append(entity_text)
                    current_entity = []
                    current_label = None
            
            return dict(entities)
            
        except Exception as e:
            logger.error(f"BERT entity extraction error: {e}")
            return {}
    
    def classifyProductTypeZeroShot(self, text: str) -> Tuple[List[str], List[float]]:
        """
            Classify product type using zero-shot classification
            
            Args:
                text (str): Input text to classify
                
            Returns:
                Tuple[List[str], List[float]]: Predicted label and confidence score
        """
        if not self.zero_shot_classifier:
            return "Unknown", 0.0
        
        try:
            text_lower = text.lower()
        
            # Detect all keyword matches
            all_matches = {}  # category -> (confidence, matched_terms)

            for main_category, synonyms in PRODUCT_DICTIONARY.items():
                all_terms = [main_category.lower()] + [syn.lower() for syn in synonyms]
                matched_terms = []
                
                for term in all_terms:
                    pattern = r'\b' + re.escape(term) + r's?\b'
                    if re.search(pattern, text_lower):
                        matched_terms.append(term)
                
                if matched_terms:
                    # Calculate confidence based on match quality
                    max_confidence = 0
                    for term in matched_terms:
                        conf = 0.95 if term == main_category.lower() else 0.85
                        max_confidence = max(max_confidence, conf)
                    
                    all_matches[main_category] = (max_confidence, matched_terms)

            # Step 2: Process results
            if len(all_matches) >= 2:                
                # Sort by confidence
                sorted_matches = sorted(all_matches.items(), key=lambda x: x[1][0], reverse=True)
                categories = [item[0] for item in sorted_matches]
                confidences = [item[1][0] for item in sorted_matches]
                
                return categories, confidences
            
            elif len(all_matches) == 1:
                # Single category found
                category = list(all_matches.keys())[0]
                confidence = all_matches[category][0]
                return [category], [confidence]
                
            else:
                # No keyword matches - use full zero-shot
                all_candidates = []
                term_to_category = {}
                
                for main_category, synonyms in PRODUCT_DICTIONARY.items():
                    all_candidates.append(main_category)
                    term_to_category[main_category] = main_category
                    
                    for synonym in synonyms:
                        all_candidates.append(synonym)
                        term_to_category[synonym] = main_category
                
                result = self.zero_shot_classifier(text, all_candidates)
                
                # For zero-shot results, check for multiple high-confidence predictions
                high_confidence_results = []
                for label, score in zip(result['labels'], result['scores']):
                    if score >= self.confidence_threshold:
                        category = term_to_category[label]
                        # Avoid duplicates
                        if not any(cat == category for cat, _ in high_confidence_results):
                            high_confidence_results.append((category, score))
                
                if high_confidence_results:
                    categories = [item[0] for item in high_confidence_results]
                    confidences = [item[1] for item in high_confidence_results]
                    return categories, confidences
                else:
                    # Return top result
                    best_term = result['labels'][0]
                    best_score = result['scores'][0]
                    best_category = term_to_category[best_term]
                    return [best_category], [best_score]
            
        except Exception as e:
            logger.error(f"Zero-shot classification error: {e}")
            return "Unknown", 0.0