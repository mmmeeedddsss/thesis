import json
import logging
import os
import pathlib
import pickle
from collections import OrderedDict
from threading import Thread
from time import sleep

import pandas as pd
from tqdm import tqdm

from dataset.amazon.loader import AmazonDatasetLoader
from models.nlp.KeyBERT import KeyBERTExtractor
from models.nlp.yake import YakeExtractor
from models.reco.TopicExtractorRecommender import TopicExtractorRecommender
from models.reco.recommendations import get_recommender_own
from user_study.metadata import metadata_loader

logger = logging.getLogger(__name__)

# CACHING IS DEPENDENT ON THE DATASET SPLIT
# so delete cache files if you change the split
DATA_CACHE_FOLDER = f'{pathlib.Path(__file__).parent.parent.absolute()}/cached_data'


class Recommender:
    def __init__(self, keyword_extractor, n_gram=2):
        print(f"Initializing Recommender {keyword_extractor}_{n_gram}")
        self.user_worker_mapping = {}
        self.keyword_extractor_name = keyword_extractor
        self.ngram = n_gram

        amazon_dataloader = AmazonDatasetLoader()
        if n_gram == 2:
            dataset_path, train_df = amazon_dataloader.get_processed_pandas_df()
        else:
            dataset_path, train_df = amazon_dataloader.get_processed_pandas_df_1()
        dataset_name = dataset_path.split('/')[-1].split('.')[0]
        cached_obj_name = f'user_property_map__user_study_{keyword_extractor}_{dataset_name}'
        cached_file_location = f'{DATA_CACHE_FOLDER}/{cached_obj_name}.pickle'
        if not os.path.isfile(cached_file_location):
            self.recommender_own = get_recommender_own(dataset_name, fit=True, df_to_fit=train_df,
                                                       keyword_extractor=keyword_extractor)
            with open(cached_file_location, 'wb') as handle:
                logger.info("Serializing user map")
                pickle.dump(self.recommender_own, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(cached_file_location, 'rb') as handle:
                self.recommender_own: TopicExtractorRecommender = pickle.load(handle)

    def generate_recommendations_async(self, user_id, blocking):
        if user_id in self.user_worker_mapping:
            try:
                self.user_worker_mapping[user_id].terminate()
            except:
                pass
        self.user_worker_mapping[user_id] = RecommendationGeneratorWorker(user_id, self.recommender_own,
                                                                          self.keyword_extractor_name, self.ngram)
        self.user_worker_mapping[user_id].start(blocking)

    def get_recommendations_of(self, user_id, blocking=False):
        filename = f'{self.keyword_extractor_name}_{self.ngram}_{user_id}_recommendations.json'
        print(filename)
        if os.path.isfile(filename):
            with open(filename, 'r') as f:
                recommendations = json.load(f)
            return json.dumps(recommendations)
        else:
            self.generate_recommendations_async(user_id, blocking)
            return '[]'


class RecommendationGeneratorWorker:
    def __init__(self, user_id, recommender_own, keyword_extractor, ngram):
        self.user_id = user_id
        self.recommender_own = recommender_own
        self.keyword_extractor_name = keyword_extractor
        self.ngram = ngram
        self.thread = None

    def start(self, blocking=False):
        if not blocking:
            thread = Thread(target=self.__do_generate)
            thread.start()
            self.thread = thread
        else:
            self.__do_generate()

    def __do_generate(self):
        url = UserReviewLoader(self.user_id, self.recommender_own)
        recommendations = url.get_top_n_recommendation(20, self.keyword_extractor_name, self.ngram)
        for reco in recommendations['recommendations']:
            print(reco['asin'])
            reco['item_metadata'] = metadata_loader.get_item(reco['asin'])
        filename = f'{self.keyword_extractor_name}_{self.ngram}_{self.user_id}_recommendations.json'
        with open(filename, 'w') as f:
            json.dump(recommendations, f)

    def terminate(self):
        pass


class UserReviewLoader:
    n_max_rows = 10000000

    def __init__(self, user_id, recommender_own: TopicExtractorRecommender):
        self.__df = None
        self.filename = f'{pathlib.Path(__file__).parent.absolute()}' \
                        f'/user_study_reviews.json'
        self.recommender_own = recommender_own
        self.user_id = user_id

    @property
    def df(self):
        if self.__df is None:
            # filename is set on extending classes
            self.__df = self._get_df_multiline_json([self.filename])
            self.__df = self.__df[
                (self.__df["user_id"].astype(str).str.contains(self.user_id, case=False))
            ]
            self.dataset_stats()

            self.__df = self.__df.rename(columns=
            {
                'user_id': 'userID',
                'asin': 'itemID',
                'comment': 'review'
            })

        return self.__df

    def dataset_stats(self):
        # Statistical summary of dataset
        print(self.df)

    def _get_df_multiline_json(self, filenames):
        dfs = []
        for filename in filenames:
            dfs.append(pd.read_json(filename, lines=True, nrows=self.n_max_rows))
        return pd.concat(dfs)

    def get_top_n_recommendation(self, n=20, keyword_extractor_name='bert', ngrams=2):
        print(f'Starting to create recommendations using {keyword_extractor_name}')
        if keyword_extractor_name == 'yake':
            print('Using Yake')
            YakeExtractor().extract_keywords(self.df)
        else:
            print('Using KeyBERT')
            KeyBERTExtractor().extract_keywords(self.df, {'top_n': 7, 'keyphrase_ngram_range': (1, ngrams)})
        property_map = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], }
        for _, row in self.df.iterrows():
            topics = [(x[1], x[0]) for x in row['topics_YakeExtractor' if keyword_extractor_name == 'yake' else 'topics_KeyBERTExtractor']]
            r = row['rating']
            property_map[r] += topics

        filtered_property_map = {}
        for rating in range(0, 6):
            if len(property_map[rating]) > 0:
                property_map[rating] = sorted(property_map[rating], reverse=True)  # True for bert
                property_map[rating] = [x[1] for x in property_map[rating]]
                property_map[rating] = list(OrderedDict.fromkeys(property_map[rating]))[:10]
                property_map[rating] = [tuple(e.split(' ')) for e in property_map[rating]]
            else:
                del property_map[rating]

        users_interests = property_map
        print(f"User interests: {users_interests}")

        item_ids = self.recommender_own.get_top_n_recommendations_for_user(user_interests=users_interests, n=n * 800)
        item_recommendations = []
        recommended_based_on = {}
        for item_id in tqdm(item_ids):
            explanations = self.recommender_own.explain_api(users_interests,
                                                            self.recommender_own.item_property_map[item_id],
                                                            explain=False, return_dict=True)
            if len(explanations) >= 1:
                flag_to_recommend = False
                for explanation in explanations:
                    if explanation['users_matching_interest'] not in recommended_based_on:
                        recommended_based_on[explanation['users_matching_interest']] = set()
                    if len(recommended_based_on[explanation['users_matching_interest']]) < 2:
                        flag_to_recommend = True
                    recommended_based_on[explanation['users_matching_interest']].add(item_id)

                if flag_to_recommend:
                    item_recommendations.append(
                        {
                            'asin': item_id,
                            'explanations': explanations,
                        }
                    )
            if len(item_recommendations) >= 16:
                break

        print('Rates of w2v hits:', self.recommender_own.rates)

        return {'recommendations': item_recommendations, 'users_interests': users_interests}


recommenders = {
    'bert': Recommender('bert'),
    'bert_1': Recommender('bert', 1),
    'yake': Recommender('yake'),
}
