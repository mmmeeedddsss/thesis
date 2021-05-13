import surprise
from sklearn.tree import DecisionTreeClassifier
from surprise import AlgoBase
from surprise import Dataset
from surprise.model_selection import cross_validate
import numpy as np
import pandas as pd
from surprise import PredictionImpossible
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from models.nlp.KeyBERT import KeyBERTExtractor
from models.nlp.yake import YakeExtractor
from gensim.models import Word2Vec
import logging


class TopicExtractorRecommender:

    def __init__(self):
        self.df = None

    def fit(self, train_df):

        logging.info("Extracting Keywords")
        self.df = YakeExtractor().extract_keywords(train_df)

        sentences = self.df['review'].tolist()
        sentences = [x.split(' ') for x in sentences]

        logging.info("Training w2v")
        self.w2v_model = Word2Vec(sentences=sentences, epochs=50, vector_size=75, window=5, min_count=1, workers=4)

        self.user_property_map = {}
        self.item_property_map = {}

        logging.info("Initializing user-item maps")
        for _, row in train_df.iterrows():
            topics = [x[0] for x in row['topics']]

            i = row['itemID']
            u = row['userID']
            r = row['rating']

            if u not in self.user_property_map:
                self.user_property_map[u] = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], }
            if i not in self.item_property_map:
                self.item_property_map[i] = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], }

            self.item_property_map[i][r] += (topics)
            self.user_property_map[u][r] += (topics)

        lr_X = []
        lr_y = []

        logging.info("Creating lr_X and lr_y")
        for _, row in train_df.iterrows():
            user_interests = self.user_property_map[row['userID']]
            item_features = self.item_property_map[row['itemID']]

            d = 0
            for rating, interests in user_interests.items():
                for _, features in item_features.items():  # omitting item ratings for its features ????
                    for interest in interests:
                        d += - abs(rating/5 - np.mean(self.w2v_model.wv.distances(interest, features)))

            d /= len(user_interests)

            lr_X.append(d)
            lr_y.append(row['rating'])

        lr_X = np.asarray(lr_X, dtype=np.float32).reshape(-1, 1)
        lr_y = np.array(lr_y).ravel()

        logging.info("Training Decision Tree")
        self.lr_model = DecisionTreeClassifier(random_state=0).fit(lr_X, lr_y)

        return self

    def estimate(self, u, i):
        user_interests = self.user_property_map[i]
        item_features = self.item_property_map[u]

        d = 0
        for rating, interests in user_interests.items():
            for _, features in item_features.items():  # omitting item ratings for its features ????
                for interest in interests:
                    d += - abs(rating / 5 - np.mean(self.w2v_model.wv.distances(interest, features)))

        d /= len(user_interests)

        return self.lr_model.predict(d)

    def accuracy(self, df):
        for i in range(5):
            logging.info(f'------------------ {i} ------------------')
            test = df.groupby('userID', as_index=False).nth(i)
            test_indexes = test.index
            train = df.loc[set(df.index) - set(test_indexes)]

            logging.info('Starting model fit')
            self.fit(train)

            mae = 0
            logging.info('Starting test')
            for _, row in test.iterrows():
                est = self.estimate(row['userID'], row['itemID'])
                mae += abs(est - test['rating'])

            mae /= len(test)

            logging.error(f'MAE FOR ITER {i}:', mae)

