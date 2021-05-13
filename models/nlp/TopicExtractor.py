

class TopicExtractor:

    def __init__(self):
        self.model = None  # set by implementing classes

    def extract_keywords_of_items(self, df):
        print(df)
        idf = df.groupby('itemID', as_index=False).agg({'review': ' '.join})
        print(idf)
        return self.extract_keywords(idf)

    def extract_keywords_of_users(self, df):
        udf = df.groupby('userID', as_index=False).agg({'review': ' '.join})
        return self.extract_keywords(udf)

    def extract_keywords(self, df):
        df['topics'] = df['review']
        df['topics'] = df['topics'].apply(lambda x: self.model.extract_keywords(x, top_n=10))
        return df
