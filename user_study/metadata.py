import pathlib

import numpy
import pandas as pd


class MetadataLoader:
    n_max_rows = 100

    def __init__(self):
        self.__df = None
        self.filename = f'{pathlib.Path(__file__).parent.parent.absolute()}' \
                        f'/dataset/amazon/meta_CDs_and_Vinyl.json'
        self.df

    @property
    def df(self):
        if self.__df is None:
            # filename is set on extending classes
            self.__df = self._get_df_multiline_json([self.filename])
            self.dataset_stats()
            self.__df = self.__df.drop(['tech1', 'tech2', 'feature', 'also_buy', 'also_view',
                                        'date', 'main_cat', 'price', 'similar_item', 'fit', 'imageURL'], axis=1)
            self.__df = self.__df[self.__df['imageURLHighRes'].map(lambda d: len(d)) > 0]
            self.__df = self.__df[self.__df['description'].map(lambda d: len(d)) > 0]

        return self.__df

    def dataset_stats(self):
        # Statistical summary of dataset
        print(self.df)

    def _get_df_multiline_json(self, filenames):
        dfs = []
        for filename in filenames:
            print(filename)
            dfs.append(pd.read_json(filename, lines=True, nrows=self.n_max_rows))
        return pd.concat(dfs)

    def get_random_row(self):
        return self.df.sample(n=1) \
            .to_json(orient="records")

    def get_item(self, item_asin):
        return self.df[
            self.df["asin"].str.contains(item_asin, case=False)
        ].to_json(orient="records")

    def search(self, q):
        x = self.df[
            (self.df["description"].astype(str).str.contains(q, case=False)) |
            (self.df["title"].str.contains(q, case=False))
            ]
        x = x.sample(n=min(len(x.index), 100)).to_json(orient="records")
        return x


metadata_loader = MetadataLoader()
