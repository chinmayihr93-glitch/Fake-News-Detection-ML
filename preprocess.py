import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download resources
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Clean text by removing special characters, stopwords and lemmatizing."""
    text = re.sub('[^a-zA-Z]', ' ', str(text)).lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)