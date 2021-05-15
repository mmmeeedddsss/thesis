import json
import math
import os
import pickle
import re

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
from gensim.models import Word2Vec, FastText
import logging

import gensim.downloader as api

logger = logging.getLogger(__name__)

DATA_CACHE_FOLDER = 'cached_data'


def serialize_param_dict(params, prefix=''):
    sparams = json.dumps(params, sort_keys=True)
    return prefix + re.sub(r'{|}|\"| ', '', sparams) \
        .replace(':', '_') \
        .replace(',', '__')


class TopicExtractorRecommender:

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.train_df = None
        self.user_property_map = None
        self.item_property_map = None

    def get_default_params(self):
        return {
            'train_test_split': {
                'max_group_size': 500,
            },
            'topic_extraction': {
                'extracted_topic_col': 'topics_KeyBERTExtractor',
            },
            'word_vectorizer': {
                'model': {
                    'epochs': 100,
                    'vector_size': 250,
                    'window': 5,
                }
            },
            'user_item_maps_generation': {
                'num_features_in_dicts': 6,
            },
            'score_rating_mapper_model': {
            },
        }

    def calculate_score(self, user_interests, item_features):
        score = [0, 0, 0, 0, 0, 0]
        for rating, interests in user_interests.items():
            for interest in interests:
                for _, features in item_features.items():  # omitting item ratings for its features ????
                    distance = np.mean(self.w2v_model.wv.distances(interest, features))
                    # old score:
                    # d += - abs(rating / 5 - np.mean(self.w2v_model.wv.distances(interest, features)))
                    # new one:
                    # avg_rating = 2.5
                    # d += (rating-avg_rating) * (1-distance)
                    score[rating] += distance

            score[rating] /= len(interests)

        return score

    def convert_score_to_x(self, score):
        return score[1:]

    def _generate_keywords(self, params):
        extracted_topic_col = params['extracted_topic_col']
        if extracted_topic_col:
            logger.info(f'Using previously generated {extracted_topic_col} col for topics')
            self.train_df['topics'] = self.train_df[extracted_topic_col]
        else:
            logger.info("Extracting Keywords")
            self.train_df = YakeExtractor().extract_keywords(self.train_df)
            self.train_df['topics'] = self.train_df['topics_YakeExtractor']

    def _train_word_vectorizer(self, params):
        sentences = self.train_df['review'].tolist()
        sentences = [x.split(' ') for x in sentences]

        cached_model_name = serialize_param_dict(params, prefix=f'word2vec__{self.dataset_name}')

        if not os.path.isfile(f'{DATA_CACHE_FOLDER}/{cached_model_name}'):
            logger.info("Training w2v")
            self.w2v_model = Word2Vec(sentences=sentences, min_count=1, workers=6, **params['model'])
            self.w2v_model.save(f'{DATA_CACHE_FOLDER}/{cached_model_name}.model')
        else:
            logger.info("Loading w2v from cache")
            self.w2v_model = Word2Vec.load(f'{DATA_CACHE_FOLDER}/word2vec__{cached_model_name}.model')

    def _generate_user_item_maps(self, params):
        def generate_map_from(col_name):
            property_map = {}
            for _, row in self.train_df.iterrows():
                topics = [(x[1], x[0]) for x in row['topics']]

                o = row[col_name]
                r = row['rating']

                if o not in property_map:
                    property_map[o] = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], }

                property_map[o][r] += topics

            for _, object_map in property_map.items():
                for rating in range(0, 6):
                    if len(object_map[rating]) > 0:
                        object_map[rating] = sorted(object_map[rating], reverse=True)[:params['num_features_in_dicts']]
                        object_map[rating] = [x[1] for x in object_map[rating]]
                    else:
                        del object_map[rating]

            return property_map
        # ---------------------------------------------------

        # For user
        cached_obj_name = serialize_param_dict(params, prefix=f'user_property_map__{self.dataset_name}')
        if not os.path.isfile(f'{DATA_CACHE_FOLDER}/{cached_obj_name}.pickle'):
            logger.info("Initializing user map")
            self.user_property_map = generate_map_from('userID')
            with open(f'{DATA_CACHE_FOLDER}/{cached_obj_name}.pickle', 'wb') as handle:
                logger.info("Serializing user map")
                pickle.dump(self.user_property_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            logger.info("Loading user map from cache")
            with open(f'{DATA_CACHE_FOLDER}/{cached_obj_name}.pickle', 'rb') as handle:
                self.user_property_map = pickle.load(handle)

        # For item
        cached_obj_name = serialize_param_dict(params, prefix=f'item_property_map__{self.dataset_name}')
        if not os.path.isfile(f'{DATA_CACHE_FOLDER}/{cached_obj_name}.pickle'):
            logger.info("Initializing item map")
            self.item_property_map = generate_map_from('itemID')
            with open(f'{DATA_CACHE_FOLDER}/{cached_obj_name}.pickle', 'wb') as handle:
                logger.info("Serializing item map")
                pickle.dump(self.item_property_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            logger.info("Loading item map from cache")
            with open(f'{DATA_CACHE_FOLDER}/{cached_obj_name}.pickle', 'rb') as handle:
                self.item_property_map = pickle.load(handle)

    def _train_score_rating_mapper(self, params):
        lr_X = []
        lr_y = []

        logger.info("Creating lr_X and lr_y")
        for _, row in tqdm(self.train_df.iterrows(), total=len(self.train_df)):
            user_interests = self.user_property_map[row['userID']]
            item_features = self.item_property_map[row['itemID']]

            score = self.calculate_score(user_interests, item_features)
            x = self.convert_score_to_x(score)

            lr_X.append(x)
            lr_y.append(row['rating'])

        lr_X = np.asarray(lr_X, dtype=np.float32)
        lr_y = np.array(lr_y).ravel()

        logger.info(f'Training set size: {len(lr_X)}')

        logger.info("Training Decision Tree")
        tree_param = {'criterion': ['gini', 'entropy'],
                      'min_samples_split': [2, 5, 15]}

        self.lr_model = GridSearchCV(DecisionTreeClassifier(), tree_param, cv=5, n_jobs=4, scoring='balanced_accuracy')
        self.lr_model = self.lr_model.fit(lr_X, lr_y)

    def fit(self, train_df, params):

        self.train_df = train_df
        self._generate_keywords(params['topic_extraction'])
        self._train_word_vectorizer(params['word_vectorizer'])
        self._generate_user_item_maps(params['user_item_maps_generation'])
        self._train_score_rating_mapper(params['score_rating_mapper_model'])

        return self

    def baseline_5(self, u, i):
        # mean int of train ratings
        return 5

    def estimate(self, u, i):
        try:
            user_interests = self.user_property_map[u]
            item_features = self.item_property_map[i]
        except:
            return None

        score = self.calculate_score(user_interests, item_features)
        x = self.convert_score_to_x(score)
        X = np.asarray(x, dtype=np.float32).reshape(1, -1)

        return self.lr_model.predict(X)

    def balance_test_set(self, test, params):
        min_group = params['max_group_size'] * 3
        for i in range(1, 6):
            min_group = min(min_group, len(test[test["rating"] == i]))

        min_group = int(min_group * 0.33)

        return pd.concat([test[test["rating"] == 1][:min_group],
                          test[test["rating"] == 2][:min_group],
                          test[test["rating"] == 3][:min_group],
                          test[test["rating"] == 4][:min_group],
                          test[test["rating"] == 5][:min_group]])

    def accuracy(self, df, params):
        for i in range(1):
            logger.info(f'------------------ {i} ------------------')
            # test = df.groupby('userID', as_index=False).nth(i)
            df = df.sample(frac=1)

            test = self.balance_test_set(df, params['train_test_split'])

            test_indexes = test.index
            train = df.loc[set(df.index) - set(test_indexes)]

            logger.info('Starting model fit')
            logger.info(f'Train set size: {len(train)}')
            logger.info(f'Test set size: {len(test)}')
            options = ['topics_KeyBERTExtractor', 'topics_YakeExtractor', None]
            self.fit(train, params)

            mae = 0
            mse = 0
            baseline_mse = 0
            baseline_mae = 0
            l = 0
            logger.info('Starting test')
            for _, row in test.iterrows():
                est = self.estimate(row['userID'], row['itemID'])
                if est is not None:
                    mae += abs(int(est) - row['rating'])
                    mse += (int(est) - row['rating']) ** 2

                    baseline_mae += abs(int(self.baseline_5(_, _)) - row['rating'])
                    baseline_mse += (int(self.baseline_5(_, _)) - row['rating']) ** 2

                    l += 1

            mae /= l
            mse /= l

            baseline_mse /= l
            baseline_mae /= l

            logger.error(f'Able to generate recommendations for {l} cases({l / len(test)})')
            print(f'MAE for iter {i}: {mae}')
            print(f'MSE for iter {i}: {mse}')
            print(f'RMSE for iter {i}: {math.sqrt(mse)}')
            logger.error(f'-----------------------------------------')
            print(f'Baseline MAE for iter {i}: {baseline_mae}')
            print(f'Baseline MSE for iter {i}: {baseline_mse}')
            print(f'Baseline RMSE for iter {i}: {math.sqrt(baseline_mse)}')
