import pandas as pd


class DatasetLoader:
    n_max_rows = 100000

    def load_data(self):
        self.df = self._get_df_multiline_json(self.filename)
        self.dataset_stats()

    def dataset_stats(self):
        # Statistical summary of dataset
        print(self.df)

    def _get_df_multiline_json(self, filename):
        return pd.read_json(filename, lines=True, nrows=self.n_max_rows)

    def read_recommender_data(self):
        raise NotImplementedError()

    def read_review_data(self):
        raise NotImplementedError()
