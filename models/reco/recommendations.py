from surprise import accuracy, SVD
from surprise.model_selection import KFold, GridSearchCV, cross_validate

from models.reco.TopicExtractorRecommender import TopicExtractorRecommender


def baseline_optimization_recommendation(data, recommender):
    param_grid = {'n_factors': [100], 'n_epochs': [30], 'lr_all': [0.01],
                  'reg_all': [0.02]}
    gs = GridSearchCV(recommender, param_grid, measures=['rmse', 'mae'], cv=3, joblib_verbose=True, n_jobs=6)

    gs.fit(data)

    # best RMSE score
    print('RMSE Train:', gs.best_score['rmse'], 'MAE Train:', gs.best_score['mae'])

    # n_factors=100, n_epochs=30, lr_all=0.01, reg_all=0.02
    params = gs.best_params['rmse']
    svdtuned = SVD(n_factors=params['n_factors'], n_epochs=params['n_epochs'], lr_all=params['lr_all'],
                   reg_all=params['reg_all'])

    print(f"n_factors={params['n_factors']}, n_epochs={params['n_epochs']}, "
          f"lr_all={params['lr_all']}, reg_all={params['reg_all']}")

    return svdtuned


def SVD_model_evaluate(data, recommender):
    kf = KFold(n_splits=3)
    for trainset, testset in kf.split(data):
        # train and test algorithm.
        recommender.fit(trainset)

        balanced_testset = []
        for i in range(1, 6):
            balanced_testset += list(filter(lambda x: x[2] == i, testset))[:500]

        predictions = recommender.test(balanced_testset)

        accuracy.mae(predictions, verbose=True)
        accuracy.mse(predictions, verbose=True)
        accuracy.rmse(predictions, verbose=True)


def get_recommender_own(dataset_name, fit=False, df_to_fit=None, override_keyword_extractor=None) -> TopicExtractorRecommender:
    params = get_default_params()
    if override_keyword_extractor:
        params['topic_extraction']['extracted_topic_col'] = \
            'topics_KeyBERTExtractor' if override_keyword_extractor == 'bert' else 'topics_YakeExtractor'
        params['user_item_maps_generation']['high_score_better'] = \
            True if override_keyword_extractor == 'bert' else False
    recommender = TopicExtractorRecommender(dataset_name, params)
    if fit:
        recommender.fit(df_to_fit, get_default_params())
    return recommender


def baseline_recommendation_own(dataset_name, df):
    recommender = get_recommender_own(dataset_name, fit=False, df_to_fit=df)
    recommender.accuracy(df, get_default_params())

    #recommender.score_map_no_predict(df, get_default_params())


def get_default_params():
    use_yake = False
    return {
        'train_test_split': {
            'max_group_size': 500,
        },
        'topic_extraction': {
            'extracted_topic_col':
                'topics_YakeExtractor' if use_yake else 'topics_KeyBERTExtractor',  # topics_KeyBERTExtractor_1-2gram, topics_KeyBERTExtractor
        },
        'word_vectorizer': {
            'model': {
                'epochs': 100,
                'vector_size': 250,
                'window': 5,
            },
            'ngram_n': 1, # don't change
        },
        'user_item_maps_generation': {
            'num_features_in_dicts': 8, # this was 15 on the digital music
            'high_score_better': False if use_yake else True,  # True for bert & tfidf, false for yake
        },
        'score_rating_mapper_model': {
        },
        'tf-idf': {
            'enabled': True,
            'use_row': 'topics_YakeExtractor' if use_yake else 'topics_KeyBERTExtractor',
        }
    }
