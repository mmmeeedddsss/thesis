import csv
import json
import pathlib
import sys
import time
from typing import Tuple

import math
import os
import pickle
import re
from collections import OrderedDict

import mmh3
from sklearn import tree
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from gensim.models import Word2Vec
import logging

import gensim.downloader as api

from models.reco.OrdinalClassifier import OrdinalClassifier

logger = logging.getLogger(__name__)

# CACHING IS DEPENDENT ON THE DATASET SPLIT
# so delete cache files if you change the split
DATA_CACHE_FOLDER = f'{pathlib.Path(__file__).parent.parent.parent.absolute()}/cached_data'

UNIQUE_WORD = 'hggf1fmasd2hb1a2dyawn1asdy21awe2nsd'


class TopicExtractorRecommender:

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
        self.lr_model2 = None
        self.ordinal_model = OrdinalClassifier()
        self.imputer = None

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

        self.unbiased_freq_dict = {}
        self.upper_unbiased_freq = 0

        self.class_weights = {False: 150, True: 1}

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

    all_features = set()


    def find_two_smallest(self, l):
        s1 = 1
        s2 = 1
        for e in l:
            if e < s2:
                if e < s1:
                    s2 = s1
                    s1 = e
                else:
                    s2 = e
        return [s1, s2]

    # also change explain method, shares logic
    def calculate_score(self, user_interests, item_features, verbose=False):
        score = [[], [], ]
        all_positive_interests = []
        all_item_features = []

        for interest_rating, interests in user_interests.items():
            if len(interests) == 0 or interest_rating < 4:
                continue
            for interest in interests:
                if self.can_use_words_in_explanation(interest):
                    all_positive_interests.append(interest)

        for _, features in item_features.items():
            for feature in features:
                if self.can_use_words_in_explanation(feature):
                    all_item_features.append(feature)

        dists = []

        for feature in all_item_features:
            for interest in all_positive_interests:
                pair_distance = self._calculate_distance(interest, feature)
                dists.append(pair_distance)

        print(dists)

        """
        for i in range(6):
            score[i].sort()
            score[i] = np.mean((score[i] + [1, 1])[:2])
        """
        score[1] = np.mean(self.find_two_smallest(dists))

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
        # mean_iidf = (1 / self.idf_mean_review) * self.INVERSE_IDF_SCALING_CONSTANT
        all_dists = []
        for interest_rating, interests in user_interests.items():
            if interest_rating < 3:
                continue

            for interest in interests:
                for feature_rating, features in item_features.items():  # omitting item ratings for its features ????
                    for feature in features:
                        interest_idf = self._get_idf_weight_reviews(interest, interest_rating)  # to scale a bit
                        feature_idf = self._get_idf_weight_reviews(feature, feature_rating)
                        pair_distance = self._calculate_distance(interest, feature, skipw1=True)
                        pair_distance_sqr = pair_distance * pair_distance
                        if verbose:
                            print(f'({interest}, {feature}, {interest_rating}) '
                                  f'-> '
                                  f'pair_distance={pair_distance:<5}, interest_idf={interest_idf:<5}, feature_idf={feature_idf:<5} ')

                        """if not self.MEAN_IIDF_LOWERBOUND <= interest_idf <= self.MEAN_IIDF_UPPERBOUND:
                            print(f'Not using \t{interest_idf:<8}:\t{interest}')
                        if not self.MEAN_IIDF_LOWERBOUND <= feature_idf <= self.MEAN_IIDF_UPPERBOUND:
                            print(f'Not using \t{feature_idf:<8}:\t{feature}')"""

                        if 0.4 > pair_distance:
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
                            f'feature_idf={feature_idf:<5}')
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
            return ret

        return len(all_dists) >= 2

    def _get_idf_weight_reviews(self, word: Tuple[str], rating):
        return np.mean([self.__get_idf_weight_reviews(w, rating) for w in word])

    def xxd(self, w):
        return self.__get_idf_weight_reviews(w, -1)

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

    rates = [0, 0, 0, 0]

    def _calculate_distance_words(self, word1, word2, skipw1=False):
        # first try to calc distance on pretrained model, else go with the custom one
        # TODO check if the individual values are in the ranges we want
        # TODO self.pretrained_w2v.distance(word1, word2) might be 0

        if (skipw1 or self.can_use_word_in_explanation(word1)) and self.can_use_word_in_explanation(word2):

            try:
                self.rates[0] += 1
                return self.pretrained_w2v.distance(word1, word2)
            except:
                try:
                    self.rates[1] += 1
                    return self.pretrained_w2v_2.distance(word1, word2)
                except:
                    try:
                        self.rates[2] += 1
                        return self.w2v_model.wv.distance(word1, word2)
                    except:
                        self.rates[3] += 1
                        return 1
        return 1

    def _calculate_distance(self, word1, word2, skipw1=False):
        d = []
        for w1 in word1:
            for w2 in word2:
                dist = self._calculate_distance_words(w1, w2, skipw1)
                if dist < 1:
                    d.append(dist)
        return np.mean(d) if len(d) else 1

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

        cached_obj_name = self.serialize_param_dict(params, prefix=f'tfidf_{self.dataset_name}')
        cached_file_location = f'{DATA_CACHE_FOLDER}/{cached_obj_name}.pickle'
        if not os.path.isfile(cached_file_location):
            logger.info("Initializing tfidf")
            self.tfidf_review.fit(self.train_df['review'].append(
                pd.Series([UNIQUE_WORD, UNIQUE_WORD, UNIQUE_WORD]), ignore_index=True))
            with open(cached_file_location, 'wb') as handle:
                logger.info("Serializing tfidf")
                pickle.dump(self.tfidf_review, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            logger.info("Loading tfidf from cache")
            with open(cached_file_location, 'rb') as handle:
                self.tfidf_review = pickle.load(handle)

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

    def can_use_words_in_explanation(self, words):
        for w in words:
            if self.can_use_word_in_explanation(w):
                return True
        return False

    def can_use_word_in_explanation(self, w):
        w_iidf = self.__get_idf_weight_reviews(w, -1)
        if w_iidf > self.lower_bound_biased and w in self.unbiased_freq_dict and self.unbiased_freq_dict[
            w] < self.upper_unbiased_freq:
            return True  # Dont filter if at least one word satisfies condition
        return False

    def _generate_word_commonity_thresholds(self):
        P_lower_biased = 98
        P_upper_unbiased = 97

        with open('dataset/unigram_freq.csv', mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file, )
            for row in csv_reader:
                self.unbiased_freq_dict[row['word']] = int(row['freq'])

        unbiased_freqs = [v for k, v in self.unbiased_freq_dict.items()]

        self.upper_unbiased_freq = np.percentile(unbiased_freqs, P_upper_unbiased)

        iidfs = [1 / v for v in self.tfidf_review.idf_]

        self.lower_bound_biased = np.percentile(iidfs, P_lower_biased)

    def _train_word_vectorizer(self, params):
        """logger.info('Importing glove vectors :')
        # https://nlp.stanford.edu/projects/glove/
        with open('cached_data/glove.6B.200d.txt', 'r') as f:
            for line in tqdm(f):
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], 'float32')
                self.glove_dict[word] = vector"""

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
        cached_model_name = self.serialize_param_dict(params, prefix=f'word2vec__{self.dataset_name}')
        # TODO change this one obviously xd
        cached_model_name = 'word2vec__CDs_and_Vinyl_1gram_combined_05model_epochs_100__vector_size_250__window_5__ngram_n_12047887974'
        cached_file_location = f'{DATA_CACHE_FOLDER}/{cached_model_name}.model'
        # TODO check https://code.google.com/archive/p/word2vec/ for accuracy test of word2vec
        if not os.path.isfile(cached_file_location):
            logger.info("Training w2v")
            sentences = self.train_df['review'].tolist()
            sentences = [x.split() for x in sentences]

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
                        object_map[rating] = list(filter(self.can_use_words_in_explanation, object_map[rating]))
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

                if x[0] != 1:
                    lr_X.append(x)
                    lr_y.append(row['rating'] >= 4)

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

        print(f"Num rows: {len(lr_X)}, num empty rows: {((lr_X == 1).astype(int).sum(axis=1) == 3).sum()}")

        self.lr_model = DecisionTreeClassifier(random_state=42, class_weight=self.class_weights)
        self.lr_model = self.lr_model.fit(lr_X, lr_y)

        self.lr_model2 = DummyClassifier(random_state=42, strategy='stratified')
        self.lr_model2 = self.lr_model2.fit(lr_X, lr_y)

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
        print('Generate word commonities')
        self._generate_word_commonity_thresholds()
        self.update_state_hash('2')
        print('user item maps are being generated')
        self._generate_user_item_maps(params['user_item_maps_generation'])
        self.update_state_hash('20_003')
        print('The best ML model')
        self._train_score_rating_mapper(params['score_rating_mapper_model'])
        return self

    def baseline_5(self, u, i):
        # mean int of train ratings
        return 1

    def estimate(self, u, i, verbose=False):
        try:
            user_interests = self.user_property_map[u]
            item_features = self.item_property_map[i]
        except:
            return None, None, None

        return self.estimate_api(user_interests, item_features, verbose)

    def estimate_api(self, user_interests, item_features, verbose=False):
        score = self.calculate_score(user_interests, item_features, verbose=verbose)

        x = self.convert_score_to_x(score)
        X = np.asarray(x, dtype=np.float32).reshape(1, -1)
        #X = self.imputer.transform(X)

        return X, self.lr_model.predict(X), self.lr_model2.predict(X)

    def get_top_n_recommendations_for_user(self, *, user_interests, n):
        print('get_top_n_recommendations_for_user')
        dists = []
        for item_asin, item_features in tqdm(self.item_property_map.items()):
            score, est, _ = self.estimate_api(user_interests, item_features)
            if est >= 4:
                dists.append((np.mean(list(filter(lambda x: x > 0.0001, score[1:]))), item_asin))

        dists.sort()
        dists = dists[:n]

        return [x[1] for x in dists]

    def balance_test_set(self, test, params):
        self.update_state_hash(f'{params["max_group_size"]}')
        min_group = params['max_group_size'] * 4
        test = test.drop_duplicates(subset=['review'])
        for i in range(1, 6):
            min_group = min(min_group, len(test[test["rating"] == i]))

        min_group = int(min_group) ## TODO ADD * 0.25

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

            #test = df[:7500]

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
            pred_ys = []
            pred_ys_2 = []
            true_ys = []
            sample_weights = []

            logger.info('Starting test')
            for _, row in test.iterrows():
                score, est, est_2 = self.estimate(row['userID'], row['itemID'])
                #print(f'{score} -> real={row["rating"]}, est= {est[0]}, est2= {est_2[0]}')
                print(score[0][0], row["rating"], est[0])
                if est is not None or score[0][0] != 1:
                    real_y = row['rating'] >= 4
                    dist = np.mean(score)
                    all_dists.append(dist)

                    mae += abs(int(est[0]) - real_y)
                    mse += (int(est[0]) - real_y) ** 2

                    baseline_mae += abs(int(self.baseline_5(_, _)) - real_y)
                    baseline_mse += (int(self.baseline_5(_, _)) - real_y) ** 2

                    l += 1

                    pred_ys.append(est[0])
                    true_ys.append(real_y)
                    pred_ys_2.append(est_2[0])
                    sample_weights.append(self.class_weights[real_y])

            def my_confusion_matrix(y_trues, y_preds):
                true = {True: 0, False: 0}
                false = {True: 0, False: 0}
                for i in range(len(y_trues)):
                    y_true = y_trues[i]
                    y_pred = y_preds[i]
                    if y_true == y_pred:
                        true[y_pred] += 1
                    else:
                        false[y_pred] += 1

                print(f'Total Positive: {(true[True] + false[False])}')
                print(f'Total Negative: {len(y_trues)-(true[True] + false[False])}')
                print(f'True Positive: {true[True]}')
                print(f'True Negative: {true[False]}')
                print(f'False Positive: {false[True]}')
                print(f'False Negative: {true[False]}')


            print('Classifier 1:')
            print(classification_report(y_true=true_ys, y_pred=pred_ys))
            print(my_confusion_matrix(y_trues=true_ys, y_preds=pred_ys))
            print(confusion_matrix(y_true=true_ys, y_pred=pred_ys))
            #print(tree.export_text(self.lr_model))
            print(f'Number of recommend signals {len(list(filter(lambda x: x, pred_ys)))}/{len(pred_ys)}')

            print('Classifier 2:')
            print(classification_report(y_true=true_ys, y_pred=pred_ys_2))
            print(my_confusion_matrix(y_trues=true_ys, y_preds=pred_ys_2))
            print(f'Number of recommend signals {len(list(filter(lambda x: x, pred_ys_2)))}/{len(pred_ys_2)}')

            print('during estimation with testset')

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

            exit(1)

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

    # --------------------------------------------------------

    def fit_no_predict(self, train_df, params):
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
        print('Generate word commonities')
        self._generate_word_commonity_thresholds()
        self.update_state_hash('2')
        print('user item maps are being generated')
        self._generate_user_item_maps(params['user_item_maps_generation'])
        return self

    def score_map_no_predict(self, df, params):
        print(df)


        logger.info(f'------------------ BALANCED ------------------')
        # test = df.groupby('userID', as_index=False).nth(i)
        df = df.sample(frac=1, random_state=42)
        self.reset_state_hash(f'42,{0}')

        #test = df[:15000]

        test = self.balance_test_set(df, params['train_test_split'])

        test_indexes = test.index
        train = df.loc[set(df.index) - set(test_indexes)]

        self.calc_corr(train, test, params, 'balanced')



        logger.info(f'------------------ RANDOM SAMPLE ------------------')
        # test = df.groupby('userID', as_index=False).nth(i)
        self.reset_state_hash(f'42,{0}')

        test = df[:25000]

        #test = self.balance_test_set(df, params['train_test_split'])

        test_indexes = test.index
        train = df.loc[set(df.index) - set(test_indexes)]

        self.calc_corr(train, test, params, 'random')

    def calc_corr(self, train, test, params, exp_name):
        logger.info('Starting model fit')
        logger.info(f'Train set size: {len(train)}')
        logger.info(f'Test set size: {len(test)}')

        self.fit_no_predict(train, params)

        logger.info('Starting test')
        ctime = int(time.time()*1000)
        with open(f'exp_{ctime}_{exp_name}', 'w') as f:
            for _, row in test.iterrows():
                try:
                    user_interests = self.user_property_map[row['userID']]
                    item_features = self.item_property_map[row['itemID']]
                except:
                    continue

                score = self.calculate_score(user_interests, item_features)

                x = self.convert_score_to_x(score)
                if x[0] != 1:
                    print(x[0], row["rating"])
                    f.write(f'{x[0]} {row["rating"]}\n')
            f.flush()
