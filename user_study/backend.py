import json

from flask import Flask, request, send_from_directory, Response

from user_study.metadata import metadata_loader
from user_study.recommender import recommenders

import logging

logger = logging.getLogger(__name__)

logging.basicConfig(
    #filename='music_2gram.out',
    #filemode='a',
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

app = Flask(__name__)
app.logger.setLevel(logging.INFO)


# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route('/')
def hello_world():
    return send_from_directory('', '', 'index.html')


@app.route('/data_sample', methods=['GET'])
def data_sample():
    x = metadata_loader.get_random_row()
    return x


@app.route('/get_item', methods=['GET'])
def get_item():
    item_asin = request.args.get('asin')
    x = metadata_loader.get_item(item_asin)
    return x


@app.route('/test', methods=['GET'])
def test():
    r = recommenders['bert'].recommender_own
    print(r.lower_bound_biased, r.upper_unbiased_freq, r.unbiased_freq_dict['ambrosia'])
    print(r.xxd('ambrosia'))

    d = {}
    for word, idx in r.tfidf_review.vocabulary_.items():
        weight = r.tfidf_review.idf_[idx]
        d[word] = 1/weight

    with open('idf_values_test.txt', 'a+') as f:
        f.write(str(d))
        f.flush()

    return Response(status=200)

@app.route('/test2', methods=['GET'])
def test2():
    p = request.args.get('q')
    print(eval(p))
    return Response(status=200)

@app.route('/user_review', methods=['POST'])
def user_review():
    data = request.form.to_dict()
    print(data)
    with open("user_study/user_study_reviews.json", "a") as f:
        f.write(json.dumps(data) + '\n')
    user_id = data['user_id']
    topic_extractor = data['recommender']
    recommenders[topic_extractor].generate_recommendations_async(user_id)
    return Response(status=200)

@app.route('/recommendations_review', methods=['POST'])
def recommendations_review():
    data = request.form.to_dict()
    print(data)
    with open("user_study/recommendations_review.json", "a") as f:
        f.write(json.dumps(data) + '\n')
    return Response(status=200)


@app.route('/search')
def search():
    return send_from_directory('', '', 'search.html')


@app.route('/search_api', methods=['GET'])
def search_api():
    search_phrase = request.args.get('q')
    return metadata_loader.search(search_phrase)


@app.route('/recommend')
def recommend():
    return send_from_directory('', '', 'recommend.html')


@app.route('/recommend_api')
def recommend_api():
    user_id = request.args.get('user_id')
    topic_extractor = request.args.get('recommender')
    return recommenders[topic_extractor].get_recommendations_of(user_id)


@app.route('/submit_user_evaluation', methods=['POST'])
def submit_user_evaluation():
    # save score
    pass


# main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    app.run()

