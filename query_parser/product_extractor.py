from transformers import pipeline
import torch, re, logging
from functools import lru_cache
from typing import Dict, List, Tuple
from config.constants import FURNITURE_TYPE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class ProductExtractor:
    def __init__(self):
        self.confidence_threshold = 0.3
        self._preparePatterns()
        self._classifier_loaded = False
        self._zero_shot_model = None
        self._heavy_model = None
        
        # preload lightweight model at startup
        _ = self.classificationPipeline  

    def _preparePatterns(self):
        self.category_patterns = {}
        self.term_to_category = {}
        for main_category, synonyms in FURNITURE_TYPE.items():
            all_terms = [main_category.lower()] + [syn.lower() for syn in synonyms]
            patterns = []
            for term in all_terms:
                pattern = re.compile(r'\b' + re.escape(term) + r's?\b', re.IGNORECASE)
                patterns.append((term, pattern, main_category))
                self.term_to_category[term] = main_category
            self.category_patterns[main_category] = patterns

    @lru_cache(maxsize=1000)
    def classifyProductType(self, text: str) -> Tuple[List[str], List[float]]:
        if not text or not text.strip():
            return ["Unknown"], [0.0]

        text_lower = text.lower().strip()
        category_matches = self._matchProductKeywords(text_lower)

        if category_matches:
            sorted_matches = sorted(category_matches.items(), key=lambda x: x[1][0], reverse=True)
            categories = [item[0] for item in sorted_matches]
            confidences = [item[1][0] for item in sorted_matches]
            return categories, confidences

        return self._zeroShotClassification(text)

    def _matchProductKeywords(self, text_lower: str) -> Dict:
        category_matches = {}
        for main_category, patterns in self.category_patterns.items():
            matched_terms, max_conf = [], 0
            for term, pattern, category in patterns:
                if pattern.search(text_lower):
                    matched_terms.append(term)
                    conf = 0.95 if term == main_category.lower() else 0.85
                    max_conf = max(max_conf, conf)
            if matched_terms:
                category_matches[main_category] = (max_conf, matched_terms)
        return category_matches

    @property
    def classificationPipeline(self):
        if not self._classifier_loaded:
            try:
                self._zero_shot_model = pipeline(
                    "zero-shot-classification",
                    model="valhalla/distilbart-mnli-12-1",  # 🔹 smaller model
                    device=0 if self._hasGpu() else -1
                )
                logger.info("✅ Lightweight zero-shot classifier loaded")
            except Exception as e:
                logger.warning(f"Could not load zero-shot classifier: {e}")
                self._zero_shot_model = None
            finally:
                self._classifier_loaded = True
        return self._zero_shot_model

    def _heavyPipeline(self):
        if self._heavy_model is None:
            logger.info("⚠️ Loading heavy fallback model (bart-large-mnli)...")
            self._heavy_model = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self._hasGpu() else -1
            )
        return self._heavy_model

    def _hasGpu(self):
        try:
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _zeroShotClassification(self, text: str) -> Tuple[List[str], List[float]]:
        if not self.classificationPipeline:
            return ["Unknown"], [0.0]

        # ✅ include both main categories and synonyms
        candidate_labels = []
        label_to_main = {}  # mapping from synonym → main category

        for main_category, synonyms in FURNITURE_TYPE.items():
            candidate_labels.append(main_category)
            label_to_main[main_category] = main_category
            for syn in synonyms:
                candidate_labels.append(syn)
                label_to_main[syn] = main_category

        truncated_text = text[:200]

        result = self.classificationPipeline(truncated_text, candidate_labels)

        # collect confident results
        high_conf_results = []
        for label, score in zip(result["labels"], result["scores"]):
            if score >= self.confidence_threshold:
                main_cat = label_to_main.get(label, label)  # map synonym → main category
                high_conf_results.append((main_cat, score))

        if high_conf_results:
            # group by main category (pick max score if multiple synonyms matched)
            grouped = {}
            for cat, score in high_conf_results:
                grouped[cat] = max(grouped.get(cat, 0), score)

            # sort by confidence
            sorted_results = sorted(grouped.items(), key=lambda x: x[1], reverse=True)
            categories, confidences = zip(*sorted_results)
            return list(categories), list(confidences)

        # ⚠️ fallback to heavy model if nothing passed threshold
        result = self._heavyPipeline()(truncated_text, candidate_labels)
        main_cat = label_to_main.get(result["labels"][0], result["labels"][0])
        return [main_cat], [result["scores"][0]]
