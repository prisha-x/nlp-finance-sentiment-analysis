import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def clean_text(text):
    """Lowercase, remove punctuation and numbers."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text):
    """Split text into word tokens."""
    return text.split()


def remove_stopwords(tokens):
    """Remove common stopwords from token list."""
    return [t for t in tokens if t not in stop_words]


def lemmatize(tokens):
    """Lemmatize tokens — reduces words to base form."""
    return [lemmatizer.lemmatize(t) for t in tokens]


def preprocess(text):
    """Full pipeline: clean → tokenize → remove stopwords → lemmatize."""
    text = clean_text(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return ' '.join(tokens)