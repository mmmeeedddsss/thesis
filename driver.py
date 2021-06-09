from surprise import Reader, Dataset, SVD, accuracy

from dataset.amazon.loader import AmazonDatasetLoader
from dataset.yelp.loader import YelpDatasetLoader

from sklearn.decomposition import LatentDirichletAllocation

from models.nlp.yake import YakeExtractor
from models.reco.recommendations import baseline_recommendation_own, \
    baseline_optimization_recommendation, SVD_model_evaluate
import pandas as pd
from surprise.model_selection import train_test_split as suprise_train_test_split

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


def train_test_split(df):
    df = df.sample(frac=1, random_state=42)

    min_group = 500
    for i in range(1, 6):
        min_group = min(min_group, len(df[df["rating"] == i]))

    min_group = int(min_group * 0.33)

    test = pd.concat([df[df["rating"] == 1][:min_group],
                      df[df["rating"] == 2][:min_group],
                      df[df["rating"] == 3][:min_group],
                      df[df["rating"] == 4][:min_group],
                      df[df["rating"] == 5][:min_group]])

    test_indexes = test.index
    train = df.loc[set(df.index) - set(test_indexes)]

    return train, test


def SVD_driver(df):
    train, test = train_test_split(df)

    reader = Reader(rating_scale=(1, 5))
    train_dataset = Dataset.load_from_df(train[['userID', 'itemID', 'rating']], reader)
    test_dataset = Dataset.load_from_df(test[['userID', 'itemID', 'rating']], reader)

    trained_svd = baseline_optimization_recommendation(train_dataset, SVD)

    _, testset = suprise_train_test_split(test_dataset, test_size=0.999999999)
    trainset, _ = suprise_train_test_split(train_dataset, test_size=0.00000001)

    trained_svd.fit(trainset)
    predictions = trained_svd.test(testset)

    accuracy.rmse(predictions=predictions, verbose=True)
    accuracy.mae(predictions=predictions, verbose=True)


if __name__ == '__main__':
    amazon_dataloader = AmazonDatasetLoader()
    yelp_dataloader = YelpDatasetLoader()

    #df = YakeExtractor().extract_keywords_of_items(amazon_dataloader.get_pandas_df())
    #print(df)

    df = amazon_dataloader.get_processed_pandas_df()

    #SVD_driver(df)

    dataset_name = amazon_dataloader.filenames[0].split('/')[-1].split('.')[0]
    baseline_recommendation_own(dataset_name, df)

    # [200 rows x 12 columns]
    # KeyBERT 124.91261499999999 secs | Yake 1.6692570000000018 secs
