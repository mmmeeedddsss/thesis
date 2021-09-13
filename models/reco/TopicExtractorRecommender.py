import json
import math
import os
import pickle
import re
from collections import OrderedDict

import mmh3
import surprise
from sklearn import tree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from surprise import AlgoBase
from surprise import Dataset
from surprise.model_selection import cross_validate
import numpy as np
import pandas as pd
from surprise import PredictionImpossible
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from tqdm import tqdm
from pathlib import Path
from models.nlp.KeyBERT import KeyBERTExtractor
from models.nlp.yake import YakeExtractor
from gensim.models import Word2Vec, FastText
import logging

import gensim.downloader as api

from models.reco.OrdinalClassifier import OrdinalClassifier

logger = logging.getLogger(__name__)

# CACHING IS DEPENDENT ON THE DATASET SPLIT
# so delete cache files if you change the split
DATA_CACHE_FOLDER = 'cached_data'


class TopicExtractorRecommender:

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.train_df = None
        self.user_property_map = None
        self.item_property_map = None

        # Model that maps the rating distance values to predicted ratings
        # takes input that is dependent to many things. A hacky solution was
        # having an hash value that all dependent places changes to see if we can
        # use saved lr_X and lr_y values.
        self.state_hash = '42'

        self.lr_model = None
        self.ordinal_model = OrdinalClassifier()
        self.imputer = None

        self.tfidf = None

        Path(DATA_CACHE_FOLDER).mkdir(parents=True, exist_ok=True)

    def reset_state_hash(self, new_state):
        self.state_hash = str(mmh3.hash(new_state, signed=False))

    def update_state_hash(self, added_state):
        self.state_hash = str(mmh3.hash(self.state_hash + str(mmh3.hash(added_state, signed=False)), signed=False))

    def serialize_param_dict(self, params, prefix=''):
        sparams = json.dumps(params, sort_keys=True)
        return prefix + re.sub(r'{|}|\"| ', '', sparams) \
            .replace(':', '_') \
            .replace(',', '__') + self.state_hash

    def get_default_params(self):
        return {
            'train_test_split': {
                'max_group_size': 600,
            },
            'topic_extraction': {
                'extracted_topic_col':
                    'topics_KeyBERTExtractor',
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
                'high_score_better': True,  # True for bert & tfidf, false for yake
            },
            'score_rating_mapper_model': {
            },
            'tf-idf': {
                'enabled': True,
            }
        }

    def calculate_score(self, user_interests, item_features):
        score = [0, 0, 0, 0, 0, 0]
        for rating, interests in user_interests.items():
            for interest in interests:
                for _, features in item_features.items():  # omitting item ratings for its features ????
                    dists = []
                    for feature in features:
                        dists.append(self._calculate_distance(interest, feature))
                    distance = np.mean(dists)
                    # old score:
                    # d += - abs(rating / 5 - np.mean(self.w2v_model.wv.distances(interest, features)))
                    # new one:
                    # avg_rating = 2.5
                    # d += (rating-avg_rating) * (1-distance)
                    score[rating] += distance

            score[rating] /= len(interests)

        for i in range(6):
            score[i] = np.mean(score)

        return score


    idf_cache = {}
    # Returns 1/idf since the smaller distance means better correlation
    def _get_idf_weight(self, word):
        try:
            if word not in self.idf_cache:
                idx = self.tfidf.vocabulary_[word]
                weight = self.tfidf.idf_[idx]
                self.idf_cache[word] = 1/weight
        except KeyError:
            weight = self.idf_mean * self.idf_mean
            self.idf_cache[word] = 1/weight
        return self.idf_cache[word]

    def _calculate_distance(self, word1, word2):
        weight1 = self._get_idf_weight(word1)
        weight2 = self._get_idf_weight(word2)

        # first try to calc distance on pretrained model, else go with the custom one
        # TODO check if the individual values are in the ranges we want
        # TODO self.pretrained_w2v.distance(word1, word2) might be 0
        try:
            return self.pretrained_w2v.distance(word1, word2) * weight1 * weight1 * weight2 * weight2
        except:
            return self.w2v_model.wv.distance(word1, word2) * weight1 * weight1 * weight2 * weight2

    def convert_score_to_x(self, score):
        return score[1:]

    def _generate_keywords(self, params):
        extracted_topic_col = params['extracted_topic_col']
        if extracted_topic_col:
            logger.info(f'Using previously generated {extracted_topic_col} col for topics')
            self.train_df['topics'] = self.train_df[extracted_topic_col]
            self.update_state_hash(extracted_topic_col)
        else:
            logger.info("Extracting Keywords")
            self.train_df = YakeExtractor().extract_keywords(self.train_df)
            self.train_df['topics'] = self.train_df['topics_YakeExtractor']
            self.update_state_hash('topics_YakeExtractor')

    def _train_tf_idf(self, params):
        if not params['enabled']:
            return

        self.tfidf = TfidfVectorizer()
        self.tfidf.fit(self.train_df['review'])
        self.idf_mean = np.mean(self.tfidf.idf_)

    def _train_word_vectorizer(self, params):
        sentences = self.train_df['review'].tolist()
        sentences = [x.split(' ') for x in sentences]

        cached_model_name = self.serialize_param_dict(params, prefix=f'word2vec__{self.dataset_name}')
        cached_file_location = f'{DATA_CACHE_FOLDER}/{cached_model_name}.model'
        if not os.path.isfile(cached_file_location):
            logger.info("Training w2v")
            self.w2v_model = Word2Vec(sentences=sentences, min_count=1, workers=6, **params['model'])
            self.w2v_model.save(cached_file_location)
        else:
            logger.info("Loading w2v from cache")
            self.w2v_model = Word2Vec.load(cached_file_location)

        self.pretrained_w2v = api.load("word2vec-google-news-300")

        self.update_state_hash(cached_model_name)

    def _generate_user_item_maps(self, params):
        self.update_state_hash('3')

        def generate_map_from(col_name):
            property_map = {}
            for _, row in self.train_df.iterrows():
                topics = [(x[1], x[0]) for x in row['topics']]

                o = row[col_name]
                r = row['rating']

                if o not in property_map:
                    property_map[o] = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], }

                property_map[o][r] += topics

            # todo think a better way to merge
            filtered_property_map = {}
            for key, object_map in property_map.items():
                num_deleted = 0
                for rating in range(0, 6):
                    if len(object_map[rating]) > 0:
                        object_map[rating] = sorted(object_map[rating],
                                                    reverse=params['high_score_better'])
                        object_map[rating] = [x[1] for x in object_map[rating]]
                        object_map[rating] = list(OrderedDict.fromkeys(object_map[rating]))[
                                             :params['num_features_in_dicts']]
                    else:
                        num_deleted += 1
                        del object_map[rating]
                if True or not num_deleted == 5:
                    filtered_property_map[key] = object_map

            return filtered_property_map

        # ---------------------------------------------------

        # For user
        cached_obj_name = self.serialize_param_dict(params, prefix=f'user_property_map__{self.dataset_name}')
        cached_file_location = f'{DATA_CACHE_FOLDER}/{cached_obj_name}.pickle'
        if not os.path.isfile(cached_file_location):
            logger.info("Initializing user map")
            self.user_property_map = generate_map_from('userID')
            with open(cached_file_location, 'wb') as handle:
                logger.info("Serializing user map")
                pickle.dump(self.user_property_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            logger.info("Loading user map from cache")
            with open(cached_file_location, 'rb') as handle:
                self.user_property_map = pickle.load(handle)

        # For item
        cached_obj_name = self.serialize_param_dict(params, prefix=f'item_property_map__{self.dataset_name}')
        cached_file_location = f'{DATA_CACHE_FOLDER}/{cached_obj_name}.pickle'
        if not os.path.isfile(cached_file_location):
            logger.info("Initializing item map")
            self.item_property_map = generate_map_from('itemID')
            with open(cached_file_location, 'wb') as handle:
                logger.info("Serializing item map")
                pickle.dump(self.item_property_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            logger.info("Loading item map from cache")
            with open(cached_file_location, 'rb') as handle:
                self.item_property_map = pickle.load(handle)

        self.update_state_hash(cached_obj_name)

    def _train_score_rating_mapper(self, params):
        self.update_state_hash('4')
        lr_X = []
        lr_y = []

        cached_file_location = f'{DATA_CACHE_FOLDER}/rating_predictor_x_y_{self.state_hash}.pickle'
        if not os.path.isfile(cached_file_location):
            logger.info("Creating lr_X and lr_y")
            for _, row in tqdm(self.train_df.iterrows(), total=len(self.train_df)):
                try:
                    user_interests = self.user_property_map[row['userID']]
                    item_features = self.item_property_map[row['itemID']]
                except KeyError:  # key removed because it has not enough features
                    pass

                score = self.calculate_score(user_interests, item_features)
                x = self.convert_score_to_x(score)

                lr_X.append(x)
                lr_y.append(row['rating'])

            self.imputer = SimpleImputer(missing_values=0, strategy='mean')
            lr_X = self.imputer.fit_transform(lr_X)

            lr_X = np.asarray(lr_X, dtype=np.float32)
            lr_y = np.array(lr_y).ravel()

            with open(cached_file_location, 'wb') as handle:
                logger.info("Serializing lr_X and lr_y")
                pickle.dump((lr_X, lr_y), handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            logger.info(f'Loading lr_X and lr_y from cached values with suffix {self.state_hash}')
            with open(cached_file_location, 'rb') as handle:
                lr_X, lr_y = pickle.load(handle)

        logger.info(f'Training set size: {len(lr_X)}')

        logger.info("Training Decision Tree")
        # {'criterion': 'entropy', 'min_samples_split': 5, 'splitter': 'best'}
        tree_param = {'criterion': ['gini', 'entropy'],
                      'min_samples_split': [2, 5, 15],
                      'splitter': ['best'],
                      'random_state': [1]}

        self.imputer = SimpleImputer(missing_values=0, strategy='mean')
        lr_X = self.imputer.fit_transform(lr_X)

        self.lr_model = GridSearchCV(DecisionTreeClassifier(),
                                     tree_param, cv=5, n_jobs=4, scoring='balanced_accuracy')
        self.lr_model = self.lr_model.fit(lr_X, lr_y)
        print(self.lr_model.best_params_)

        # self.ordinal_model.fit(lr_X, lr_y)

    def fit(self, train_df, params):

        self.train_df = train_df
        self._generate_keywords(params['topic_extraction'])
        self._train_word_vectorizer(params['word_vectorizer'])
        self.update_state_hash('1')
        self._generate_user_item_maps(params['user_item_maps_generation'])
        self.update_state_hash('2')
        self._train_tf_idf(params['tf-idf'])
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
            return None, None

        score = self.calculate_score(user_interests, item_features)
        x = self.convert_score_to_x(score)
        X = np.asarray(x, dtype=np.float32).reshape(1, -1)
        X = self.imputer.transform(X)

        return x, self.lr_model.predict(X)

    def estimate_ordinal(self, u, i):
        try:
            user_interests = self.user_property_map[u]
            item_features = self.item_property_map[i]
        except:
            return None

        score = self.calculate_score(user_interests, item_features)
        x = self.convert_score_to_x(score)
        X = np.asarray(x, dtype=np.float32).reshape(1, -1)
        X = self.imputer.transform(X)

        return self.lr_model.predict(X)

    def balance_test_set(self, test, params):
        self.update_state_hash(f'{params["max_group_size"]}')
        min_group = params['max_group_size'] * 3
        test = test.drop_duplicates(subset=['review'])
        for i in range(1, 6):
            min_group = min(min_group, len(test[test["rating"] == i]))

        min_group = int(min_group * 0.33)

        return pd.concat([test[test["rating"] == 5][:min_group],
                          test[test["rating"] == 1][:min_group],
                          test[test["rating"] == 2][:min_group],
                          test[test["rating"] == 3][:min_group],
                          test[test["rating"] == 4][:min_group]])

    def accuracy(self, df, params):
        for i in range(1):
            logger.info(f'------------------ {i} ------------------')
            # test = df.groupby('userID', as_index=False).nth(i)
            df = df.sample(frac=1, random_state=42)
            self.reset_state_hash(f'42,{i}')

            # test = df[:7500]

            test = self.balance_test_set(df, params['train_test_split'])

            test_indexes = test.index
            train = df.loc[set(df.index) - set(test_indexes)]

            logger.info('Starting model fit')
            logger.info(f'Train set size: {len(train)}')
            logger.info(f'Test set size: {len(test)}')

            self.fit(train, params)

            mae = 0
            mse = 0
            baseline_mse = 0
            baseline_mae = 0
            l = 0
            all_dists = []

            logger.info('Starting test')
            for _, row in test.iterrows():
                score, est = self.estimate(row['userID'], row['itemID'])
                if est is not None:
                    dist = np.mean(score)
                    all_dists.append(dist)
                    mae += abs(int(est[0]) - row['rating'])
                    mse += (int(est[0]) - row['rating']) ** 2

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

            logger.info("Explaining the recommendations")
            num_good_examples = 0
            num_bad_examples = 0
            num_total = 20

            all_dists.sort()
            for _, row in test.iterrows():
                score, est = self.estimate(row['userID'], row['itemID'])
                if est is not None:
                    dist = np.mean(score)
                    # TODO there is distance low high times est good bad cases(4 cases)
                    if int(est[0]) - row['rating'] < mae and num_good_examples < num_total and dist in all_dists[:num_total*3]:
                        user_interests = self.user_property_map[row['userID']]
                        item_features = self.item_property_map[row['itemID']]
                        logger.info(f'--------------------------------')
                        logger.info(
                            f"Generated a good prediction on the following (est: {int(est[0])}, reality:{row['rating']}):")
                        logger.info(
                            f"Similarity order is {all_dists.index(dist)}")
                        #logger.info(f'User rows: {df[df["userID"] == row["userID"]][["userID", "topics_KeyBERTExtractor", "review"]].to_string()}')
                        #logger.info(f'Item rows: {df[df["itemID"] == row["itemID"]][["itemID", "topics_KeyBERTExtractor", "review"]].to_string()}')
                        logger.info(f'User: {user_interests}')
                        logger.info(f'Item: {item_features}')
                        logger.info(f'--------------------------------')
                        logger.info('')
                        num_good_examples += 1
                    if abs(int(est[0]) - row['rating']) > mae and num_bad_examples < num_total and dist in all_dists[-num_total*3:]:
                        logger.info(f'--------------------------------')
                        user_interests = self.user_property_map[row['userID']]
                        item_features = self.item_property_map[row['itemID']]
                        logger.info(
                            f"Generated a bad prediction on following (est: {int(est[0])}, reality:{row['rating']}): ")
                        logger.info(
                            f"Similarity order is -{len(all_dists) - all_dists.index(dist)}")
                        #logger.info(f'User rows: {df[df["userID"] == row["userID"]][["userID", "topics_KeyBERTExtractor", "review"]].to_string()}')
                        #logger.info(f'Item rows: {df[df["itemID"] == row["itemID"]][["itemID", "topics_KeyBERTExtractor", "review"]].to_string()}')
                        logger.info(f'User: {user_interests}')
                        logger.info(f'Item: {item_features}')
                        logger.info(f'--------------------------------')
                        logger.info('')
                        num_bad_examples += 1
