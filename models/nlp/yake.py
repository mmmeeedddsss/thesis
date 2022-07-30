from yake import KeywordExtractor

from models.nlp.TopicExtractor import TopicExtractor


class YakeExtractor(TopicExtractor):
    def __init__(self, n=2):
        super().__init__()
        self.model = KeywordExtractor(lan="en", n=n, top=10)
