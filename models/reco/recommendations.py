from surprise import accuracy, SVD
from surprise.model_selection import KFold, GridSearchCV, cross_validate

from models.reco.TopicExtractorRecommender import TopicExtractorRecommender


def baseline_optimization_recommendation(data, recommender):
    param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],
                  'reg_all': [0.4, 0.6]}
    gs = GridSearchCV(recommender, param_grid, measures=['rmse', 'mae'], cv=3, joblib_verbose=True, n_jobs=4)

    gs.fit(data)

    # best RMSE score
    print('RMSE:', gs.best_score['rmse'], 'MAE:', gs.best_score['mae'])


def baseline_recommendation_surprise(data, recommender):
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


def baseline_recommendation_own(dataset_name, data):
    recommender = TopicExtractorRecommender(dataset_name)
    recommender.accuracy(data, recommender.get_default_params())
