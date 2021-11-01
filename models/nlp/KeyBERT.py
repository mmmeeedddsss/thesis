from keybert import KeyBERT

from models.nlp.TopicExtractor import TopicExtractor


class KeyBERTExtractor(TopicExtractor):
    def __init__(self):
        super().__init__()
        from sentence_transformers import SentenceTransformer
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.model = KeyBERT(model=sentence_model)
