import json
import pathlib
from typing import Tuple

import math
import os
import pickle
import re
from collections import OrderedDict

import mmh3
import surprise
from gensim.models.phrases import Phraser, Phrases, ENGLISH_CONNECTOR_WORDS
from scipy import spatial
from sklearn import tree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from spacy.lang.en import STOP_WORDS
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
DATA_CACHE_FOLDER = f'{pathlib.Path(__file__).parent.parent.parent.absolute()}/cached_data'

UNIQUE_WORD = 'hggf1fmasd2hb1a2dyawn1asdy21awe2nsd'


class TopicExtractorRecommender:
    INVERSE_IDF_SCALING_CONSTANT = 1.75

    def __init__(self, dataset_name, params):
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

        self.sentences = None

        ngram = params['init']['ngram_n']
        self.tfidf_review = TfidfVectorizer(ngram_range=(1, 1))
        self.idf_mean_review = None
        self.idf_min_review = None

        # not used in current setup
        self.tfidf_topics = [TfidfVectorizer(ngram_range=(1, 1)), TfidfVectorizer(ngram_range=(1, 1)),
                             TfidfVectorizer(ngram_range=(1, 1)), TfidfVectorizer(ngram_range=(1, 1)),
                             TfidfVectorizer(ngram_range=(1, 1))]  # xd
        self.idf_mean_topics = [None] * 5

        self.glove_dict = {}

        self.idf_cache = {
            0: {},  # we dont use idf per rating now, so only the 0th is used
            1: {},
            2: {},
            3: {},
            4: {},
        }

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

    # also change explain method, shares logic
    def calculate_score(self, user_interests, item_features, verbose=False):
        score = [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
        mean_iidf = (1 / self.idf_mean_review) * self.INVERSE_IDF_SCALING_CONSTANT
        for interest_rating, interests in user_interests.items():
            if len(interests) == 0:
                continue
            for interest in interests:
                interest_idf = self._get_idf_weight_reviews(interest, interest_rating)  # to scale a bit
                if interest_idf >= mean_iidf:
                    continue
                for feature_rating, features in item_features.items():  # omitting item ratings for its features ????
                    dists = []
                    for feature in features:
                        feature_idf = self._get_idf_weight_reviews(feature, feature_rating)
                        if feature_idf >= mean_iidf:
                            continue
                        pair_distance = self._calculate_distance(interest, feature)
                        pair_distance_sqr = pair_distance * pair_distance
                        if verbose:
                            print(f'({interest:<12}, {feature:<12}, {interest_rating}) '
                                  f'-> '
                                  f'pair_distance={pair_distance:<5}')
                        dists.append(pair_distance_sqr)
                    m = np.mean(dists) if len(dists) else 1
                    distance = m + np.mean((dists + [1, 1, 1])[:3])
                    score[interest_rating].append(distance)

        for i in range(6):
            score[i].sort()
            score[i] = np.mean(score[i])

        return score

    def explain(self, u, i, verbose=False, explain=False, verbose_context=False, user_rows=None, item_rows=None):
        try:
            user_interests = self.user_property_map[u]
            item_features = self.item_property_map[i]
        except:
            return None, None

        return self.explain_api(user_interests, item_features, verbose, explain, verbose_context, user_rows, item_rows)

    def explain_api(self, user_interests, item_features, verbose=False, explain=False,
                    verbose_context=False, user_rows=None, item_rows=None, return_dict=False):
        mean_iidf = (1 / self.idf_mean_review) * self.INVERSE_IDF_SCALING_CONSTANT
        all_dists = []
        for interest_rating, interests in user_interests.items():
            if interest_rating < 3:
                continue

            for interest in interests:
                for feature_rating, features in item_features.items():  # omitting item ratings for its features ????
                    for feature in features:
                        interest_idf = self._get_idf_weight_reviews(interest, interest_rating)  # to scale a bit
                        feature_idf = self._get_idf_weight_reviews(feature, feature_rating)
                        pair_distance = self._calculate_distance(interest, feature)
                        pair_distance_sqr = pair_distance * pair_distance
                        if verbose:
                            print(f'({interest}, {feature}, {interest_rating}) '
                                  f'-> '
                                  f'pair_distance={pair_distance:<5}, interest_idf={interest_idf:<5}, feature_idf={feature_idf:<5} '
                                  f'mean_iidf={mean_iidf:<5}')

                        if 0.6 > pair_distance and \
                                interest_idf <= mean_iidf and feature_idf <= mean_iidf:
                            all_dists.append((pair_distance_sqr * interest_idf * feature_idf,
                                              interest, feature, pair_distance_sqr, interest_idf, feature_idf))

        if explain:
            all_dists.sort()
            for i in range(min(3, len(all_dists))):
                interest = all_dists[i][1]
                feature = all_dists[i][2]
                pair_distance_sqr, interest_idf, feature_idf = tuple(all_dists[i][3:])
                logger.info(f'**** User mentioned {interest}, item is {feature} ***')
                logger.info(f'pair_distance_squared={pair_distance_sqr:<5}, interest_idf={interest_idf:<5}, '
                            f'feature_idf={feature_idf:<5}, mean_iidf={(1 / self.idf_mean_review) * 3 / 2}')
                if verbose_context:
                    logger.info(f"User's context of mention(s):")
                    for comment in user_rows[user_rows['review'].str.contains(interest) == True]['review'].values:
                        logger.info('--- ' + comment)
                    logger.info(f"Item's context of mention(s):")
                    for comment in item_rows[item_rows['review'].str.contains(feature) == True]['review'].values:
                        logger.info('--- ' + comment)
                    logger.info('-----------')

        if return_dict:
            ret = []
            features = {}
            all_dists.sort()
            for i in range(min(5, len(all_dists))):
                score = all_dists[i][0]
                interest = all_dists[i][1]
                feature = all_dists[i][2]
                if f"{feature}" not in features:
                    ret.append(
                        {
                            'item_feature': feature,
                            'score': score,
                            'users_matching_interest': interest,
                        }
                    )
                    features[f"{feature}"] = 1
            if len(all_dists) > 0:
                logger.info('-----------')
                logger.info('-----------')
            return ret

        return len(all_dists) >= 2

    def _get_idf_weight_reviews(self, word: Tuple[str], rating):
        return np.mean([self.__get_idf_weight_reviews(w, rating) for w in word])

    # Smaller distance means better correlation
    # Bigger idf means more unique so returning 1/idf
    def __get_idf_weight_reviews(self, word, rating):
        try:
            if word not in self.idf_cache[0]:
                idx = self.tfidf_review.vocabulary_[word]
                weight = self.tfidf_review.idf_[idx]
                self.idf_cache[0][word] = 1 / weight
        except KeyError as e:
            weight = self.idf_mean_review
            self.idf_cache[0][word] = 1 / weight
        return self.idf_cache[0][word]

    f1 = [0, 0]
    f2 = [0, 0]
    f3 = [0, 0]

    def find_similarity_glove(self, word1, word2):
        return np.dot(self.glove_dict[word1], self.glove_dict[word2])

    def _calculate_distance_words(self, word1, word2):
        # first try to calc distance on pretrained model, else go with the custom one
        # TODO check if the individual values are in the ranges we want
        # TODO self.pretrained_w2v.distance(word1, word2) might be 0

        try:
            return self.pretrained_w2v.distance(word1, word2)
        except:
            try:
                return self.pretrained_w2v_2.distance(word1, word2)
            except:
                try:
                    return self.w2v_model.wv.distance(word1, word2)
                except:
                    return 1

    def _calculate_distance(self, word1, word2):
        d = []
        for w1 in word1:
            dd = []
            for w2 in word2:
                dd.append(self._calculate_distance_words(w1, w2))
            d.append(np.min(dd))
        return np.mean(d)

    def convert_score_to_x(self, score):
        return score[1:]

    def _generate_keywords(self, params):
        extracted_topic_col = params['extracted_topic_col']
        if extracted_topic_col:
            logger.info(f'Using previously generated {extracted_topic_col} col for topics')
            print(self.train_df)
            self.train_df['topics'] = self.train_df[extracted_topic_col]
            self.update_state_hash(extracted_topic_col)
        else:
            logger.info("Extracting Keywords")
            self.train_df['topics'] = self.train_df['topics_YakeExtractor']
            self.update_state_hash('topics_YakeExtractor')

    def _train_tf_idf(self, params):
        logger.info(f'Training TF-IDF')

        self.tfidf_review.fit(self.train_df['review'].append(
            pd.Series([UNIQUE_WORD, UNIQUE_WORD, UNIQUE_WORD]), ignore_index=True))
        self.idf_mean_review = np.mean(self.tfidf_review.idf_)
        self.idf_min_review = self._get_idf_weight_reviews((UNIQUE_WORD,), -1)

        print('Min idf value is :', 1 / self.idf_min_review)

        """corpus = [None, [], [], [], [], []]
        for user_id, v in self.user_property_map.items():
            for rating, keys in v.items():
                corpus[rating].append(' '.join(keys) + ' ')
        corpus = corpus[1:]
        for i in range(5):
            self.tfidf_topics[i].fit(corpus[i])
            self.idf_mean_topics[i] = np.mean(self.tfidf_topics[i].idf_)"""

    def _train_word_vectorizer(self, params):
        """logger.info('Importing glove vectors :')
        # https://nlp.stanford.edu/projects/glove/
        with open('cached_data/glove.6B.200d.txt', 'r') as f:
            for line in tqdm(f):
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], 'float32')
                self.glove_dict[word] = vector"""

        sentences = self.train_df['review'].tolist()
        sentences = [x.split() for x in sentences]

        """if params['ngram_n'] > 1:
            ngrams = []
            for ind, list_of_words in enumerate(sentences):
                new = []
                for w in list_of_words:
                    if w not in STOP_WORDS:
                        new.append(w)

                from nltk import ngrams as ng
                ngrams.append(list(ng(new, params['ngram_n'])))

            sentences = sentences + ngrams
        """
        self.sentences = sentences

        cached_model_name = self.serialize_param_dict(params, prefix=f'word2vec__{self.dataset_name}')
        cached_file_location = f'{DATA_CACHE_FOLDER}/{cached_model_name}.model'
        # TODO check https://code.google.com/archive/p/word2vec/ for accuracy test of word2vec
        if not os.path.isfile(cached_file_location):
            logger.info("Training w2v")
            self.w2v_model = Word2Vec(sentences=sentences, min_count=1, workers=6, **params['model'])
            self.w2v_model.save(cached_file_location)
        else:
            logger.info("Loading w2v from cache")
            self.w2v_model = Word2Vec.load(cached_file_location)

        logger.info("Loading pretrained w2v vectors")
        self.pretrained_w2v = api.load("glove-twitter-200")
        self.pretrained_w2v_2 = api.load("glove-wiki-gigaword-200")

        self.update_state_hash(cached_model_name)

    def _generate_user_item_maps(self, params):
        self.update_state_hash('14')

        def generate_map_from(col_name):
            def filter_func(w):
                w_tfidf = self._get_idf_weight_reviews(w, -1)
                return self.idf_min_review <= w_tfidf <= (1 / self.idf_mean_review) * self.INVERSE_IDF_SCALING_CONSTANT

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
            for key, object_map in tqdm(property_map.items()):
                num_deleted = 0
                for rating in range(0, 6):
                    if len(object_map[rating]) > 0:
                        object_map[rating] = sorted(object_map[rating],
                                                    reverse=params['high_score_better'])
                        object_map[rating] = [x[1] for x in object_map[rating]]
                        object_map[rating] = list(OrderedDict.fromkeys(object_map[rating]))
                        object_map[rating] = [tuple(e.split(' ')) for e in object_map[rating]]
                        object_map[rating] = list(filter(filter_func, object_map[rating]))
                        object_map[rating] = object_map[rating][:params['num_features_in_dicts']]
                    else:
                        num_deleted += 1
                        del object_map[rating]
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

        # TODO ADDED TRIMMING FOR FAST ITERATIONS(5 hr to 25 mins)
        logger.info("Trimming user and item property maps")
        for user, rating_keywords in self.item_property_map.items():
            for rating, keyword in rating_keywords.items():
                rating_keywords[rating] = keyword[:8]
        for user, rating_keywords in self.user_property_map.items():
            for rating, keyword in rating_keywords.items():
                rating_keywords[rating] = keyword[:8]

        self.update_state_hash(cached_obj_name)

    def _train_score_rating_mapper(self, params):
        lr_X = []
        lr_y = []

        cached_file_location = f'{DATA_CACHE_FOLDER}/rating_predictor_{self.dataset_name}_x_y_{self.state_hash}.pickle'
        if not os.path.isfile(cached_file_location):
            logger.info("Creating lr_X and lr_y for score mapper")
            for _, row in tqdm(self.train_df.iterrows(), total=len(self.train_df)):
                try:
                    user_interests = self.user_property_map[row['userID']]
                    item_features = self.item_property_map[row['itemID']]
                except KeyError:  # key removed because it has not enough features
                    continue

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

        # self.ordinal_model.fit(lr_X, lr_y)

    def fit(self, train_df, params):
        print('Fitting ..')
        self.update_state_hash('8')
        self.train_df = train_df
        print('Generating keywords and w2v')
        self._generate_keywords(params['topic_extraction'])
        self._train_word_vectorizer(params['word_vectorizer'])
        self.update_state_hash('8')
        print('TfIdf fitting')
        self._train_tf_idf(params['tf-idf'])
        self.update_state_hash('10')
        print('user item maps are being generated')
        self._generate_user_item_maps(params['user_item_maps_generation'])
        self.update_state_hash('17')
        print('The best ML model')
        self._train_score_rating_mapper(params['score_rating_mapper_model'])

        return self

    def baseline_5(self, u, i):
        # mean int of train ratings
        return 5

    def estimate(self, u, i, verbose=False):
        try:
            user_interests = self.user_property_map[u]
            item_features = self.item_property_map[i]
        except:
            return None, None

        return self.estimate_api(user_interests, item_features, verbose)

    def estimate_api(self, user_interests, item_features, verbose=False):
        score = self.calculate_score(user_interests, item_features, verbose=verbose)

        x = self.convert_score_to_x(score)
        X = np.asarray(x, dtype=np.float32).reshape(1, -1)
        X = self.imputer.transform(X)

        return x, self.lr_model.predict(X)

    def get_top_n_recommendations_for_user(self, *, user_interests, n):
        print('get_top_n_recommendations_for_user')
        dists = []
        for item_asin, item_features in tqdm(self.item_property_map.items()):
            score, est = self.estimate_api(user_interests, item_features)
            if est >= 4:
                dists.append((np.mean(list(filter(lambda x: x > 0.0001, score[1:]))), item_asin))

        dists.sort()
        dists = dists[:n]

        return [x[1] for x in dists]

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
        print(df)
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

            print('during estimation with testset')
            print(self.f1, self.f2, self.f3, end='\n')
            self.f1 = [0, 0]
            self.f2 = [0, 0]
            self.f3 = [0, 0]

            mae /= l
            mse /= l

            baseline_mse /= l
            baseline_mae /= l

            logger.info(f'Able to generate recommendations for {l} cases({l / len(test)})')
            logger.info(f'MAE for iter {i}: {mae}')
            logger.info(f'MSE for iter {i}: {mse}')
            logger.info(f'RMSE for iter {i}: {math.sqrt(mse)}')
            logger.info(f'-----------------------------------------')
            logger.info(f'Baseline MAE for iter {i}: {baseline_mae}')
            logger.info(f'Baseline MSE for iter {i}: {baseline_mse}')
            logger.info(f'Baseline RMSE for iter {i}: {math.sqrt(baseline_mse)}')

            logger.info("Explaining the recommendations")
            num_good_examples = 0
            num_bad_examples = 0
            num_total = 50

            all_num_explanation = 0

            pd.set_option('display.max_colwidth', None)
            all_dists.sort()
            for _, row in test.iterrows():
                score, est = self.estimate(row['userID'], row['itemID'])
                if est is not None:
                    dist = np.mean(score)
                    # TODO there is distance low high times est good bad cases(4 cases)
                    all_num_explanation += int(self.explain(row['userID'], row['itemID']))
                    if int(est[0]) - row['rating'] < mae and num_total > 0 and row['rating'] > 3:

                        if self.explain(row['userID'], row['itemID']):
                            num_total -= 1 if self.explain(row['userID'], row['itemID'], explain=True, verbose=False,
                                                           user_rows=df.loc[df['userID'] == row['userID']][
                                                               ['userID', 'itemID', 'review']],
                                                           item_rows=df.loc[df['itemID'] == row['itemID']][
                                                               ['userID', 'itemID', 'review']],
                                                           verbose_context=False,
                                                           ) == 1 else 0
                        else:
                            continue

                        user_interests = self.user_property_map[row['userID']]
                        item_features = self.item_property_map[row['itemID']]
                        logger.info(f'--------------------------------')
                        logger.info(
                            f"Generated a good prediction on the following (est: {int(est[0])}, reality:{row['rating']}):")
                        logger.info(
                            f"Similarity order is {all_dists.index(dist)}")
                        # logger.info(f'User rows: {df[df["userID"] == row["userID"]][["userID", "topics_KeyBERTExtractor", "review"]].to_string()}')
                        # logger.info(f'Item rows: {df[df["itemID"] == row["itemID"]][["itemID", "topics_KeyBERTExtractor", "review"]].to_string()}')
                        logger.info(f'User: {user_interests}')
                        logger.info(f'Item: {item_features}')
                        logger.info(f'--------------------------------')
                        logger.info('')
                        num_good_examples += 1
                    if False and abs(
                            int(est[0]) - row['rating']) > mae and num_bad_examples < num_total and dist in all_dists[
                                                                                                            -num_total * 3:]:
                        logger.info(f'--------------------------------')
                        user_interests = self.user_property_map[row['userID']]
                        item_features = self.item_property_map[row['itemID']]
                        logger.info(
                            f"Generated a bad prediction on following (est: {int(est[0])}, reality:{row['rating']}): ")
                        logger.info(
                            f"Similarity order is -{len(all_dists) - all_dists.index(dist)}")
                        # logger.info(f'User rows: {df[df["userID"] == row["userID"]][["userID", "topics_KeyBERTExtractor", "review"]].to_string()}')
                        # logger.info(f'Item rows: {df[df["itemID"] == row["itemID"]][["itemID", "topics_KeyBERTExtractor", "review"]].to_string()}')
                        logger.info(f'User: {user_interests}')
                        logger.info(f'Item: {item_features}')
                        logger.info(f'--------------------------------')
                        logger.info('')
                        num_bad_examples += 1

            print(f'Able to generate explanation for {all_num_explanation}/{l} rows of test')
