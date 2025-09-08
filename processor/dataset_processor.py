# Import necessary libraries
import re
import pandas as pd
from utils.helpers import toCamelCase
from logs.logger import loggerSetup, logger
from typing import Dict, List, Tuple, Optional, Union
from config.constants import FURNITURE_CONFIG, MATERIALS, COLORS

# Configure logging
loggerSetup()

class DatasetProcessor:
    """Process furniture datasets for training"""
    
    def __init__(self, data_files: List[str]):
        """
            Initialize the dataset processor
            
            Args:
                csv_files (List[str]): List of CSV files to process
        """
        self.data_files = data_files
        self.processed_data = []
        self.entity_labels = set()
        
    def loadAndProcessDatasets(self) -> List[Dict]:
        """
            Load and process furniture datasets
            
            Returns:
                List[Dict]: Processed data with text and entities
        """
        all_data = []
        
        for files in self.data_files:
            try:
                if '.csv' in str(files):
                    df = pd.read_csv(files)

                elif '.xlsx' in str(files):
                    df = pd.read_excel(files)
                    df.columns = [toCamelCase(str(col)) for col in df.columns]  # Normalize column names                    

                # Process each row
                for _, row in df.iterrows():
                    processed_row = self._processRow(row)
                    print(f"Processed row: {processed_row}")  # Debugging output
                    if processed_row:
                        all_data.append(processed_row)
                        
            except Exception as e:
                logger.error(f"Error processing {files}: {e}")
                continue
        
        logger.info(f"Processed {len(all_data)} total records from datasets")
        return all_data
    
    def _processRow(self, row: pd.Series) -> Optional[Dict]:
        """
            Process a single row from dataset
            
            Args:
                row (pd.Series): Row data from DataFrame
                
            Returns:
                Optional[Dict]: Processed data with text and entities
        """
        try:
            # Extract common fields
            product_name = str(row.get('name', '') or row.get('productName', ''))
            description = str(row.get('product_details', '') or row.get('description', ''))
            category = str(row.get('furniture_type', '') or row.get('furnitureType', ''))

            if (not product_name or product_name == 'nan') or (not description or description == 'nan'):
                return None

            # Create training text
            training_text = f"{product_name} {description}".strip()
            
            if not training_text or training_text == 'nan':
                return None
            
            # Extract entities from product data
            entities = self._extractEntities(training_text)
            
            return {
                'text': training_text,
                'entities': entities,
                'category': category,
                'product_name': product_name
            }
            
        except Exception as e:
            logger.error(f"Error processing row: {e}")
            return None
    
    def _extractEntities(self, text: str) -> List[Tuple]:
        """
            Extract entities from training text
            
            Args:
                row (pd.Series): Row data from DataFrame
                text (str): Training text
                
            Returns:
                List[Tuple]: List of entities with start, end indices and labels
        """
        entities = []
        text_lower = text.lower()
        
        for furniture_type in FURNITURE_CONFIG:
            if furniture_type in text_lower:
                start_idx = text_lower.find(furniture_type)
                end_idx = start_idx + len(furniture_type)
                entities.append((start_idx, end_idx, 'PRODUCT'))
                self.entity_labels.add('PRODUCT')
        
        for material in MATERIALS:
            if material in text_lower:
                start_idx = text_lower.find(material)
                end_idx = start_idx + len(material)
                entities.append((start_idx, end_idx, 'MATERIAL'))
                self.entity_labels.add('MATERIAL')
        
        for color in COLORS:
            if color in text_lower:
                start_idx = text_lower.find(color)
                end_idx = start_idx + len(color)
                entities.append((start_idx, end_idx, 'COLOR'))
                self.entity_labels.add('COLOR')
        
        return entities
    
    def generateTrainingData(self) -> List[Tuple]:
        """
            Generate training data in spaCy format
            
            Returns:
                List[Tuple]: List of tuples with text and entities for training
        """
        training_data = []
        processed_data = self.loadAndProcessDatasets()
        
        for item in processed_data:
            training_data.append((item['text'], {'entities': item['entities']}))
        
        return training_data