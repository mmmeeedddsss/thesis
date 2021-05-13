from surprise import Reader, Dataset
import pathlib
from dataset.loader import DatasetLoader


class AmazonDatasetLoader(DatasetLoader):
    f1 = f'{pathlib.Path(__file__).parent.absolute()}/Digital_Music_5.json'
    f2 = f'{pathlib.Path(__file__).parent.absolute()}/Kindle_Store_5.json'
    filenames = [f1]

    def read_recommender_data(self):
        df = self.df.rename(columns={'overall': 'rating', 'reviewerID': 'userID', 'asin': 'itemID'})

        reader = Reader(rating_scale=(1, 5))
        return {
            'data': Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader),
        }

    def get_pandas_df(self):
        df = self.df.rename(columns={'overall': 'rating',
                                     'reviewerID': 'userID',
                                     'asin': 'itemID',
                                     'reviewText': 'review'})
        df = df.dropna(subset=['rating', 'userID', 'itemID', 'review'])
        df['review'] = df['review'].str.lower()
        return df
