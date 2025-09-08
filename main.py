"""
Advanced Furniture Parser AI Agent with ML/NLP Models
Enhanced system using spaCy NER, BERT, and Flipkart dataset training
"""
import os
from pathlib import Path
import asyncio

from flask_cors import CORS
from flask import Flask, request, jsonify, Response


from processor.furniture_parser import FurnitureParser
from datasets.data import downloadNltkData
from config.constants import FLIPKART_DATASET_PATH, MERGED_DATASET_PATH
from logs.logger import loggerSetup, logger

import json
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

loggerSetup()

# Initialize flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# # Ensure NLTK data is downloaded
# downloadNltkData()

async def queryParser(query: str):
    # Example of training with real data
    logger.info("="*80)
    logger.info("Starting furniture query parser")
    logger.info("="*80)

    # Initialize the dataset
    flipkart_dataset_path = Path(FLIPKART_DATASET_PATH)
    merged_dataset_path = Path(MERGED_DATASET_PATH)

    # Check if the dataset directory exists
    if not flipkart_dataset_path.exists() or not flipkart_dataset_path.is_dir():
        logger.error(f"Flipkart dataset directory does not exist: {flipkart_dataset_path}")
        use_dataset = False

    data_files = []

    flipkart_files = [f for f in flipkart_dataset_path.glob("*.*") if f.is_file()]
    merged_files = [f for f in merged_dataset_path.glob("*.*") if f.is_file()]
    
    data_files = flipkart_files + merged_files

    logger.info(f"Found {len(data_files)} dataset files to process")

    # Initialize the parser with the dataset files

    #TODO: Need to solve the issue when dataset is avaliable for training
    parser = FurnitureParser(data_files)
    parser.train_models(use_dataset=False)
    
    # Test after training
    # query = "couch with rounded arms, split back with add-on cushions, wooden slanted legs in a modern style to seat 3 people between 30000 and 40000 in Delhi"
    result = parser.inputToDict(query)
    print(f"\nTrained Model Result for: '{query}'")
    print(json.dumps(result, indent=2))
    return result

@app.route('/query/analyze', methods=['POST'])
def analyzeQuery():
    """Main processing endpoint"""
    try:
        print(request)
        query = request.json.get('query')
        print()
        
        if not query:
            return jsonify({
                "success": False,
                "error": "Query is required"
            }), 400
        
        result = asyncio.run(queryParser(query))
                        
        if not result:
            return jsonify({
                "success": False,
                "error": "Invalid input type"
            }), 400
        
        return jsonify({
            "success": True,
            "result": result,
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}"
        }), 500

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    logger.info("Frontend should be accessible at: http://localhost:8000")
    logger.info("API endpoints:")
    logger.info("  GET  / - Home page")
    logger.info("  POST /process - Process form data")
    
    app.run(debug=True, host='0.0.0.0', port=8000)