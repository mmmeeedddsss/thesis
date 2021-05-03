from surprise.model_selection import KFold, GridSearchCV
from surprise import SVD, accuracy

from dataset.amazon.loader import AmazonDatasetLoader
from dataset.yelp.loader import YelpDatasetLoader


def baseline_optimization_recommendation(data, recommender):
    param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],
                  'reg_all': [0.4, 0.6]}
    gs = GridSearchCV(recommender, param_grid, measures=['rmse', 'mae'], cv=3, joblib_verbose=True, n_jobs=4)

    gs.fit(data['data'])

    # best RMSE score
    print('RMSE:', gs.best_score['rmse'], 'MAE:', gs.best_score['mae'])

    # combination of parameters that gave the best RMSE score
    print(gs.best_params['mae'])


def baseline_recommendation(data, recommender):
    kf = KFold(n_splits=3)
    for trainset, testset in kf.split(data['data']):
        # train and test algorithm.
        recommender.fit(trainset)
        predictions = recommender.test(testset)

        accuracy.mae(predictions, verbose=True)


if __name__ == '__main__':
    # _data = AmazonDatasetLoader(AmazonDatasetLoader.filename).read_recommender_data()
    _data = YelpDatasetLoader(YelpDatasetLoader.filename).read_recommender_data()
    baseline_optimization_recommendation(_data, SVD)
