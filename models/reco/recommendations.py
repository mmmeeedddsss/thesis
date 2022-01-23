from surprise import accuracy, SVD
from surprise.model_selection import KFold, GridSearchCV, cross_validate

from models.reco.TopicExtractorRecommender import TopicExtractorRecommender


def baseline_optimization_recommendation(data, recommender):
    param_grid = {'n_factors': [50, 100, 150, 250], 'n_epochs': [10, 20, 30], 'lr_all': [0.005, 0.01],
                  'reg_all': [0.02, 0.1]}
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


def get_recommender_own(dataset_name, fit=False, df_to_fit=None) -> TopicExtractorRecommender:
    recommender = TopicExtractorRecommender(dataset_name, get_default_params())
    if fit:
        recommender.fit(df_to_fit, get_default_params())
    return recommender


def baseline_recommendation_own(dataset_name, data):
    recommender = get_recommender_own(dataset_name, fit=False)
    recommender.accuracy(data, get_default_params())


def get_default_params():
    return {
        'init': {
            'ngram_n': 2
        },
        'train_test_split': {
            'max_group_size': 600,
        },
        'topic_extraction': {
            'extracted_topic_col':
                'topics_KeyBERTExtractor_1-2gram', # topics_KeyBERTExtractor_1-2gram, topics_KeyBERTExtractor
        },
        'word_vectorizer': {
            'model': {
                'epochs': 100,
                'vector_size': 250,
                'window': 5,
            },
            'ngram_n': 1,
        },
        'user_item_maps_generation': {
            'num_features_in_dicts': 15,
            'high_score_better': True,  # True for bert & tfidf, false for yake
        },
        'score_rating_mapper_model': {
        },
        'tf-idf': {
            'enabled': True,
            'use_row': 'topics_KeyBERTExtractor',
        }
    }
