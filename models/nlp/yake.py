from yake import KeywordExtractor

from models.nlp.TopicExtractor import TopicExtractor


class YakeExtractor(TopicExtractor):
    def __init__(self):
        super().__init__()
        self.model = KeywordExtractor(lan="en", n=2, top=10)
