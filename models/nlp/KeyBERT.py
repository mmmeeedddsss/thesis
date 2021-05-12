from keybert import KeyBERT

from models.nlp.TopicExtractor import TopicExtractor


class KeyBERTExtractor(TopicExtractor):
    def __init__(self):
        super().__init__()
        self.model = KeyBERT('distilbert-base-nli-mean-tokens')