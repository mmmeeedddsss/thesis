import surprise
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from surprise import AlgoBase
from surprise import Dataset
from surprise.model_selection import cross_validate
import numpy as np
import pandas as pd
from surprise import PredictionImpossible
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from tqdm import tqdm

from models.nlp.KeyBERT import KeyBERTExtractor
from models.nlp.yake import YakeExtractor
from gensim.models import Word2Vec
import logging

NUM_FEATURES_IN_DICTS = 5


class TopicExtractorRecommender:

    def __init__(self):
        self.df = None

    def fit(self, train_df):

        logging.info("Extracting Keywords")
        self.df = YakeExtractor().extract_keywords(train_df)

        sentences = self.df['review'].tolist()
        sentences = [x.split(' ') for x in sentences]

        logging.info("Training w2v")
        self.w2v_model = Word2Vec(sentences=sentences, epochs=250, vector_size=100, window=5, min_count=1, workers=4)

        self.user_property_map = {}
        self.item_property_map = {}

        logging.info("Initializing user-item maps")
        for _, row in train_df.iterrows():
            topics = [(x[1], x[0]) for x in row['topics']]

            i = row['itemID']
            u = row['userID']
            r = row['rating']

            if u not in self.user_property_map:
                self.user_property_map[u] = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], }
            if i not in self.item_property_map:
                self.item_property_map[i] = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], }

            self.item_property_map[i][r] += (topics)
            self.user_property_map[u][r] += (topics)

        for _, user_map in self.user_property_map.items():
            for rating in range(0, 6):
                if len(user_map[rating]) > 0:
                    user_map[rating] = sorted(user_map[rating], reverse=True)[:NUM_FEATURES_IN_DICTS]
                    user_map[rating] = [x[1] for x in user_map[rating]]
                else:
                    del user_map[rating]

        for _, item_map in self.item_property_map.items():
            for rating in range(0, 6):
                if len(item_map[rating]) > 0:
                    item_map[rating] = sorted(item_map[rating], reverse=True)[:NUM_FEATURES_IN_DICTS]
                    item_map[rating] = [x[1] for x in item_map[rating]]
                else:
                    del item_map[rating]

        lr_X = []
        lr_y = []

        logging.info("Creating lr_X and lr_y")
        for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
            user_interests = self.user_property_map[row['userID']]
            item_features = self.item_property_map[row['itemID']]

            d = 0
            for rating, interests in user_interests.items():
                for interest in interests:
                    for _, features in item_features.items():  # omitting item ratings for its features ????
                        d += - abs(rating / 5 - np.mean(self.w2v_model.wv.distances(interest, features)))

                d /= len(interests)

            lr_X.append(d)
            lr_y.append(row['rating'])

        lr_X = np.asarray(lr_X, dtype=np.float32).reshape(-1, 1)
        lr_y = np.array(lr_y).ravel()

        logging.info(f'Training set size: {len(lr_X)}')

        logging.info("Training Decision Tree")
        tree_param = {'criterion': ['gini', 'entropy'],
                      'min_samples_split': [2, 5, 15]}

        self.lr_model = GridSearchCV(DecisionTreeClassifier(min_samples_split=1), tree_param, cv=5, n_jobs=4)
        self.lr_model = self.lr_model.fit(lr_X, lr_y)

        return self

    def estimate(self, u, i):
        try:
            user_interests = self.user_property_map[u]
            item_features = self.item_property_map[i]
        except:
            return None

        d = 0
        for rating, interests in user_interests.items():
            for interest in interests:
                for _, features in item_features.items():  # omitting item ratings for its features ????
                    d += - abs(rating / 5 - np.mean(self.w2v_model.wv.distances(interest, features)))
            d /= len(interests)

        return self.lr_model.predict(np.asarray([d], dtype=np.float32).reshape(-1, 1))

    def accuracy(self, df):
        for i in range(5):
            logging.info(f'------------------ {i} ------------------')
            test = df.groupby('userID', as_index=False).nth(i)
            test_indexes = test.index
            train = df.loc[set(df.index) - set(test_indexes)]

            logging.info('Starting model fit')
            logging.info(f'Train set size: {len(train)}')
            logging.info(f'Test set size: {len(test)}')
            self.fit(train)

            mae = 0
            l = 0
            logging.info('Starting test')
            for _, row in test.iterrows():
                est = self.estimate(row['userID'], row['itemID'])
                if est is not None:
                    mae += abs(int(est) - row['rating'])
                    l += 1

            mae /= l

            logging.error(f'Able to generate recommendations for {l} cases({l / len(test)})')
            logging.error(f'MAE FOR ITER {i}: {mae} ||||')
