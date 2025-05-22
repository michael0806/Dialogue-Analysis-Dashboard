import re
from transformers import pipeline
import spacy
import warnings

# Load spaCy model (for tokenization)
nlp = spacy.blank("en")

# Load Hugging Face sentiment pipeline with three-way sentiment model
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

FILLER_WORDS = [
    "um", "uh", "like", "you know", "actually", "basically", "so", "well"
]

warnings.filterwarnings("ignore", category=UserWarning)

def parse_transcript(filepath):
    """Parse transcript.txt into a list of dicts: [{'speaker': 'A', 'text': ...}, ...]"""
    turns = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Speaker A:"):
                speaker = "A"
                text = line[len("Speaker A:"):].strip()
            elif line.startswith("Speaker B:"):
                speaker = "B"
                text = line[len("Speaker B:"):].strip()
            else:
                continue
            turns.append({"speaker": speaker, "text": text})
    return turns

def compute_sentiment(text):
    """Return sentiment label: POSITIVE, NEGATIVE, or NEUTRAL."""
    result = sentiment_pipeline(text)[0]
    label = result["label"].upper()
    # The model returns 'LABEL_0', 'LABEL_1', 'LABEL_2' for negative, neutral, positive
    if label in ["LABEL_0", "NEGATIVE"]:
        return "NEGATIVE"
    elif label in ["LABEL_1", "NEUTRAL"]:
        return "NEUTRAL"
    elif label in ["LABEL_2", "POSITIVE"]:
        return "POSITIVE"
    else:
        return label

def compute_filler_ratio(text):
    """Compute ratio: (number of filler words) / (total words)."""
    tokens = [token.text.lower() for token in nlp(text)]
    total_words = len(tokens)
    if total_words == 0:
        return 0.0
    filler_count = 0
    for word in FILLER_WORDS:
        # Count occurrences of each filler word (as a word, not substring)
        pattern = r'\b' + re.escape(word) + r'\b'
        filler_count += len(re.findall(pattern, text.lower()))
    return filler_count / total_words

def compute_word_count(text):
    """Bonus metric: word count for each turn."""
    return len([token.text for token in nlp(text)]) 