from sklearn.feature_extraction.text import CountVectorizer
from surprise import Reader, Dataset

from dataset.loader import DatasetLoader


class YelpDatasetLoader(DatasetLoader):
    filenames = [f'dataset/yelp/yelp_academic_dataset_review.json']

    def read_recommender_data(self):
        df = self.df.rename(columns={'stars': 'rating', 'user_id': 'userID', 'business_id': 'itemID'})

        reader = Reader(rating_scale=(1, 5))
        return {
            'data': Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader),
        }

    def get_pandas_df(self):
        df = self.df.rename(columns={'stars': 'rating', 'user_id': 'userID', 'business_id': 'itemID', 'text': 'review'})
        df = df.dropna(subset=['rating', 'userID', 'itemID', 'review'])
        return df
