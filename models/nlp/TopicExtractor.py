

class TopicExtractor:

    def __init__(self):
        self.model = None  # set by implementing classes

    def extract_keywords_of_items(self, df, kwargs={}):
        idf = df.groupby('itemID', as_index=False).agg({'review': ' '.join})
        return self.extract_keywords(idf, kwargs)

    def extract_keywords_of_users(self, df, kwargs={}):
        udf = df.groupby('userID', as_index=False).agg({'review': ' '.join})
        return self.extract_keywords(udf, kwargs)

    def extract_keywords(self, df, kwargs={}):
        df['topics'] = df['review']
        df['topics'] = df['topics'].apply(lambda x: self.model.extract_keywords(x, **kwargs))
        return df
