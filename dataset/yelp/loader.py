from surprise import Reader, Dataset

from dataset.loader import DatasetLoader


class YelpDatasetLoader(DatasetLoader):
    filename = f'dataset/yelp/yelp_academic_dataset_review.json'

    n_rows = 1000000

    def read_recommender_data(self):
        df = self.df.rename(columns={'stars': 'rating', 'user_id': 'userID', 'business_id': 'itemID'})

        reader = Reader(rating_scale=(1, 5))
        return {
            'data': Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader),
        }
