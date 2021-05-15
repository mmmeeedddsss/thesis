import pandas as pd


class DatasetLoader:
    n_max_rows = 1000000000

    def __init__(self):
        self.__df = None

    @property
    def df(self):
        if self.__df is None:
            # filename is set on extending classes
            self.__df = self._get_df_multiline_json(self.filenames)
            self.dataset_stats()
        return self.__df

    def dataset_stats(self):
        # Statistical summary of dataset
        print(self.df)

    def _get_df_multiline_json(self, filenames):
        dfs = []
        for filename in filenames:
            dfs.append(pd.read_json(filename, lines=True, nrows=self.n_max_rows))
        return pd.concat(dfs)

    def read_recommender_data(self):
        raise NotImplementedError()

    def get_pandas_df(self):
        raise NotImplementedError()
