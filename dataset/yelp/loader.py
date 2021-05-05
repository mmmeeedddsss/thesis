from sklearn.feature_extraction.text import CountVectorizer
from surprise import Reader, Dataset

from dataset.loader import DatasetLoader


class YelpDatasetLoader(DatasetLoader):
    filename = f'dataset/yelp/yelp_academic_dataset_review.json'

    def read_recommender_data(self):
        df = self.df.rename(columns={'stars': 'rating', 'user_id': 'userID', 'business_id': 'itemID'})

        reader = Reader(rating_scale=(1, 5))
        return {
            'data': Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader),
        }

    def read_review_data(self):
        df = self.df.rename(columns={'text': 'review'})

        cv = CountVectorizer(max_df=0.80, min_df=10, stop_words='english')
        df = cv.fit_transform(df['review'])

        return {
            'data': df,
            'cv_features': cv.get_feature_names()
        }
