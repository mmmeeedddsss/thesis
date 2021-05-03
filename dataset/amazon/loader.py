from surprise import Reader, Dataset
from dataset.loader import DatasetLoader


class AmazonDatasetLoader(DatasetLoader):
    f1 = f'dataset/amazon/Digital_Music_5.json'
    f2 = f'dataset/amazon/Kindle_Store_5.json'
    filename = f2

    def read_recommender_data(self):
        df = self.df.rename(columns={'overall': 'rating', 'reviewerID': 'userID', 'asin': 'itemID'})

        reader = Reader(rating_scale=(1, 5))
        return {
            'data': Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader),
        }
