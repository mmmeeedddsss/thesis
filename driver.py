from dataset.amazon.loader import AmazonDatasetLoader
from dataset.yelp.loader import YelpDatasetLoader

from sklearn.decomposition import LatentDirichletAllocation

from models.nlp.yake import YakeExtractor


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

    df = YakeExtractor().extract_keywords_of_items(amazon_dataloader.get_pandas_df())
    print(df)

    # [200 rows x 12 columns]
    # KeyBERT 124.91261499999999 secs | Yake 1.6692570000000018 secs
