from surprise import Reader, Dataset
import pathlib

from tqdm import tqdm

from dataset.loader import DatasetLoader
from models.preprocessing.preprocessing import preprocessing_pipeline
import pandas as pd


class AmazonDatasetLoader(DatasetLoader):
    f1 = f'{pathlib.Path(__file__).parent.absolute()}/Movies_and_TV_5.json'
    f2 = f'{pathlib.Path(__file__).parent.absolute()}/Musical_Instruments_5.json'
    f3 = f'{pathlib.Path(__file__).parent.absolute()}/Arts_Crafts_and_Sewing_5.json'
    processed_f1 = f'{pathlib.Path(__file__).parent.absolute()}/processed/Movies_and_TV_5_with_extracted_topics.gzip'
    processed_f2 = f'{pathlib.Path(__file__).parent.absolute()}/processed/Musical_Instruments_5_with_extracted_topics.gzip'
    processed_f3 = f'{pathlib.Path(__file__).parent.absolute()}/processed/Arts_Crafts_and_Sewing_5_with_extracted_topics.gzip'
    filenames = [f1]

    def read_recommender_data(self):
        df = self.df.rename(columns={'overall': 'rating', 'reviewerID': 'userID', 'asin': 'itemID'})

        reader = Reader(rating_scale=(1, 5))
        return {
            'data': Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader),
        }

    def get_pandas_df(self):
        self.df.drop(['reviewerName', 'verified', 'reviewTime', 'unixReviewTime', 'summary'], axis=1, inplace=True)
        df = self.df.rename(columns={'overall': 'rating',
                                     'reviewerID': 'userID',
                                     'asin': 'itemID',
                                     'reviewText': 'review'})
        df = df.dropna(subset=['rating', 'userID', 'itemID', 'review'])

        tqdm.pandas()
        for op in preprocessing_pipeline:
            print(op.__name__)
            df['review'] = df['review'].progress_apply(op)

        return df

    def get_processed_pandas_df(self):
        return pd.read_pickle(self.processed_f2)
