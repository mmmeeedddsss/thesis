from sklearn.feature_extraction.text import CountVectorizer
from surprise import Reader, Dataset
import pandas as pd
from dataset.loader import DatasetLoader


class YelpDatasetLoader(DatasetLoader):
    filenames = [f'dataset/yelp/yelp_academic_dataset_review.json']
    processed_filename = 'dataset/yelp/processed/yelp_academic_dataset_review.gzip'

    def read_recommender_data(self):
        df = self.df.rename(columns={'stars': 'rating', 'user_id': 'userID', 'business_id': 'itemID'})

        reader = Reader(rating_scale=(1, 5))
        return {
            'data': Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader),
        }

    def get_pandas_df(self):
        df = self.df.rename(columns={'stars': 'rating', 'user_id': 'userID', 'business_id': 'itemID', 'text': 'review'})
        df = df.dropna(subset=['rating', 'userID', 'itemID', 'review'])
        df['review'] = df['review'].str.lower()
        return df

    def get_processed_pandas_df(self):
        return pd.read_pickle(self.processed_f2)
