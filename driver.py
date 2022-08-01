from typing import List

from sklearn.metrics import classification_report, precision_recall_curve
from surprise import Reader, Dataset, SVD, accuracy, Prediction

from dataset.amazon.loader import AmazonDatasetLoader

from sklearn.decomposition import LatentDirichletAllocation

from models.nlp.yake import YakeExtractor
from models.reco.recommendations import baseline_recommendation_own, \
    baseline_optimization_recommendation, SVD_model_evaluate
import pandas as pd
from surprise.model_selection import train_test_split as suprise_train_test_split

import logging

logger = logging.getLogger(__name__)

logging.basicConfig(
    #filename='music_2gram.out',
    #filemode='a',
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


def train_test_split(df, is_validation):
    min_group = 500 * 4
    for i in range(1, 6):
        min_group = min(min_group, len(df[df["rating"] == i]))

    min_group = int(min_group * 0.25)

    if is_validation:
        test = pd.concat([df[df["rating"] == 1][-min_group:],
                          df[df["rating"] == 2][-min_group:],
                          df[df["rating"] == 3][-min_group:],
                          df[df["rating"] == 4][-min_group:],
                          df[df["rating"] == 5][-min_group:]])
    else:
        test = pd.concat([df[df["rating"] == 1][:min_group],
                          df[df["rating"] == 2][:min_group],
                          df[df["rating"] == 3][:min_group],
                          df[df["rating"] == 4][:min_group],
                          df[df["rating"] == 5][:min_group]])

    test_indexes = test.index
    train = df.loc[set(df.index) - set(test_indexes)]

    return train, test


def SVD_driver(df, test_sample_type='balanced'):
    is_validation = True

    print(f"is_validation: {is_validation}, test_sample_type: {test_sample_type}")

    df = df.sample(frac=1, random_state=42)
    if is_validation:
        if test_sample_type == 'balanced':
            train, test = train_test_split(df, is_validation)
        else:
            test = df[-15000:]
            test_indexes = test.index
            train = df.loc[set(df.index) - set(test_indexes)]
    else:
        if test_sample_type == 'balanced':
            train, test = train_test_split(df)
        else:
            test = df[:15000]
            test_indexes = test.index
            train = df.loc[set(df.index) - set(test_indexes)]

    reader = Reader(rating_scale=(1, 5))
    train_dataset = Dataset.load_from_df(train[['userID', 'itemID', 'rating']], reader)
    test_dataset = Dataset.load_from_df(test[['userID', 'itemID', 'rating']], reader)

    trained_svd = baseline_optimization_recommendation(train_dataset, SVD)

    _, testset = suprise_train_test_split(test_dataset, test_size=0.999999999)
    trainset, _ = suprise_train_test_split(train_dataset, test_size=0.00000001)

    trained_svd.fit(trainset)
    predictions: List[Prediction] = trained_svd.test(testset)

    accuracy.rmse(predictions=predictions, verbose=True)
    accuracy.mae(predictions=predictions, verbose=True)

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
        print(f'Total Negative: {len(y_trues) - (true[True] + false[False])}')
        print(f'True Positive: {true[True]}')
        print(f'True Negative: {true[False]}')
        print(f'False Positive: {false[True]}')
        print(f'False Negative: {false[False]}')

    true_y = []
    pred_y = []

    for prediction in predictions:
        true_y.append(prediction.r_ui >= 4)
        pred_y.append(prediction.est >= 4)

    my_confusion_matrix(true_y, pred_y)
    print(classification_report(y_true=true_y, y_pred=pred_y))

    precision, recall, thresholds = precision_recall_curve(true_y,
                                                           [prediction.est/5 for prediction in predictions])

    print(str(list(precision)), '\n', str(list(recall)), '\n', str(list(thresholds)))


if __name__ == '__main__':
    amazon_dataloader = AmazonDatasetLoader()

    # df = YakeExtractor().extract_keywords_of_items(amazon_dataloader.get_pandas_df())
    # print(df)

    dataset_path, df = amazon_dataloader.get_processed_pandas_df_1()

    dataset_name = dataset_path.split('/')[-1].split('.')[0]

    #SVD_driver(df, 'balanced')
    #SVD_driver(df, 'random')

    baseline_recommendation_own(dataset_name, df)

    # [200 rows x 12 columns]
    # KeyBERT 124.91261499999999 secs | Yake 1.6692570000000018 secs
