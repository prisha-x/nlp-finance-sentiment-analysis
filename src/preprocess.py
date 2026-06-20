import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def clean_text(text):
    """
    Lowercases the text and strips out anything that isn't a letter.
    Numbers and punctuation get removed since they add noise for TF-IDF.
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text):
    """
    Splits cleaned text into individual word tokens.
    Just a simple whitespace split, nothing fancy.
    """
    return text.split()


def remove_stopwords(tokens):
    """
    Removes stopwords from tokenized text.
    Financial stopwords like 'company' are also removed since they
    appear across all sentiment classes and don't help distinguish them.
    """
    return [t for t in tokens if t not in stop_words]


def lemmatize(tokens):
    """
    Reduces each token to its base form using WordNet.
    Helps the model treat 'rose' and 'rising' as related concepts
    instead of two completely separate words.
    """
    return [lemmatizer.lemmatize(t) for t in tokens]


def preprocess(text):
    """
    Runs the full pipeline: clean → tokenize → remove stopwords → lemmatize.
    This is the function actually called in the notebooks — the others
    above are just the individual steps.
    """
    text = clean_text(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return ' '.join(tokens)