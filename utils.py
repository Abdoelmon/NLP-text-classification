# utils.py

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


try:

    lemmatizer = WordNetLemmatizer()
    english_stop_words = set(stopwords.words('english'))
except Exception as e:
    print("NLTK resources not fully loaded. Ensure you ran nltk.download() commands.")

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # URLs
    text = re.sub(r'\d+', '', text) # Numbers
    text = re.sub(r'#\w+', '', text) # Hashtags
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip() 


    tokens = word_tokenize(text)


    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in english_stop_words and word.isalpha()
    ]


    return " ".join(tokens)