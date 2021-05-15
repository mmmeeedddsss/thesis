from surprise import Reader, Dataset, SVD

from dataset.amazon.loader import AmazonDatasetLoader
from dataset.yelp.loader import YelpDatasetLoader

from sklearn.decomposition import LatentDirichletAllocation

from models.nlp.yake import YakeExtractor
from models.reco.recommendations import baseline_recommendation_own, baseline_recommendation_surprise, \
    baseline_optimization_recommendation

import logging

logger = logging.getLogger(__name__)

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def lda_on_review_comments(data):
    lda = LatentDirichletAllocation(n_components=7, random_state=42, n_jobs=6, verbose=True)
    lda.fit(data['data'])

    for index, topic in enumerate(lda.components_):
        print(f'Top 15 words for Topic #{index}')
        print([data['cv_features'][i] for i in topic.argsort()[-15:]])
        print('\n')


if __name__ == '__main__':
    amazon_dataloader = AmazonDatasetLoader()
    yelp_dataloader = YelpDatasetLoader()

    #df = YakeExtractor().extract_keywords_of_items(amazon_dataloader.get_pandas_df())
    #print(df)

    df = amazon_dataloader.get_processed_pandas_df()

    #reader = Reader(rating_scale=(1, 5))
    #dataset = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)

    #baseline_recommendation_surprise(dataset, SVD())

    baseline_recommendation_own(amazon_dataloader.filenames[0], df)

    # [200 rows x 12 columns]
    # KeyBERT 124.91261499999999 secs | Yake 1.6692570000000018 secs
