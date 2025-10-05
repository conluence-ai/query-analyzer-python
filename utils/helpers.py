# Import necessary libraries
import Levenshtein
from difflib import SequenceMatcher
from spellchecker import SpellChecker
from typing import Set, Optional, Tuple

# Spell checker
spell = SpellChecker(distance=2)

# Fuzzy matching threshold
fuzzy_threshold = 0.85

def spellCorrect(text: str) -> str:
    """
        Applies spell correction to the input text

        Args:
            text (str): The input string to be spell-checked and corrected.

        Returns:
            str: The corrected string.
    """
    corrected = []
    
    for word in text.split():
        if len(word) > 2:  # Don't correct very short words
            corrected_word = spell.correction(word)
            corrected.append(corrected_word if corrected_word else word)
        else:
            corrected.append(word)
    
    return " ".join(corrected)

def fuzzyMatch(phrase: str, candidates: Set[str], threshold: Optional[float] = None) -> Optional[Tuple[str, float]]:
    """
        Find the best fuzzy match for a phrase among candidates
            
        Args:
            phrase (str): The phrase to match
            candidates (Dict[str, str]): Dictionary mapping lowercase names to original names
            threshold (float, optional): Similarity threshold (0-1)
                
        Returns:
            Optional[Tuple[str, float]]: Tuple of (matched_name, similarity_score) or None
    """
    if threshold is None:
        threshold = fuzzy_threshold

    if not phrase or not candidates:
        return None
    
    best_match = None
    best_score = 0
        
    for candidate in candidates:
        try:
            # Use Levenshtein distance if available, otherwise SequenceMatcher
            similarity = Levenshtein.ratio(phrase.lower(), candidate.lower())
        except (ImportError, NameError):
            similarity = SequenceMatcher(None, phrase.lower(), candidate.lower()).ratio()
            
        if similarity > best_score and similarity >= fuzzy_threshold:
            best_score = similarity
            best_match = candidate
        
    return (best_match, best_score) if best_match else None