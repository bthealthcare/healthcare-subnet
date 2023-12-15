import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

# Ensure you have the necessary NLTK data
nltk.download('punkt')

def get_bleu_score(reference: str, candidate: str) -> float:
    # Tokenizing the strings
    reference_tokenized = [word_tokenize(reference.lower())]
    candidate_tokenized = [word_tokenize(candidate.lower())]

    if len(reference_tokenized) == 1 and len(candidate_tokenized) == 1:
        return 1.0 if reference_tokenized[0] == candidate_tokenized[0] else 0.0

    # Define the smoothing function
    chencherry = SmoothingFunction()

    # Calculating BLEU Score with smoothing
    score = sentence_bleu(
        reference_tokenized, 
        candidate_tokenized,
        smoothing_function=chencherry.method1  # This is one of the available smoothing methods
    )
    return score