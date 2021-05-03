import pandas as pd


class DatasetLoader:
    n_max_rows = 1000000

    def __init__(self, filename):
        self.df = self._get_df_multiline_json(filename)
        self.dataset_stats()

    def read_recommender_data(self):
        raise NotImplementedError()

    def dataset_stats(self):
        # Statistical summary of dataset
        print(self.df)

    def _get_df_multiline_json(self, filename):
        return pd.read_json(filename, lines=True, nrows=self.n_max_rows)
