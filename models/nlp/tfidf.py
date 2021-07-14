from sklearn.feature_extraction.text import TfidfVectorizer

from models.nlp.TopicExtractor import TopicExtractor


class tfidfExtractor(TopicExtractor):
    def __init__(self):
        super().__init__()
        self.model = TfidfVectorizer(stop_words='english', min_df=0.02, max_df=0.20)
        self.TOP_N = 20

    def extract_keywords(self, df, kwargs={}):
        column_name = f'topics_{self.__class__.__name__}'
        df[column_name] = df['review']

        # taken from https://kavita-ganesan.com/extracting-keywords-from-text-tfidf/
        def sort_coo(coo_matrix):
            coo_matrix = coo_matrix.tocoo()
            tuples = zip(coo_matrix.col, coo_matrix.data)
            return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)[:self.TOP_N]


        self.tfidf_x = self.model.fit_transform(df['review'])
        column_score_tuples_2d = map(sort_coo, self.tfidf_x)
        # self.tfidf_v.get_feature_names()
        feature_list = self.model.get_feature_names()

        def get_keywords(column_score_tuples_1d):
            kws = []
            for col_index, score in column_score_tuples_1d:
                kws.append((feature_list[col_index], score))
            return kws

        kws = list(map(get_keywords, column_score_tuples_2d))

        df[column_name] = kws
        return df