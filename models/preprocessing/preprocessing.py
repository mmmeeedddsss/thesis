import nltk
from nltk.stem import WordNetLemmatizer
import re

lemma = WordNetLemmatizer()
nltk.download('wordnet')


def clean_non_alphanumeric_and_nums(text):
    return re.sub('[^a-zA-Z0-9]', ' ', text)


def to_lower(text):
    return str(text).lower()


def clean_lemmatization(text):
    v = ' '.join([lemma.lemmatize(word=w, pos='v') for w in text.split(' ')])
    return ' '.join([lemma.lemmatize(word=w, pos='a') for w in v.split(' ')])


def trim_whitespaces(text):
    return re.sub(' +', ' ', text).strip(' ')


preprocessing_pipeline = [
    to_lower,
    clean_non_alphanumeric_and_nums,
    clean_lemmatization,
    trim_whitespaces,
]
