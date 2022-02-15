import json

from flask import Flask, request, send_from_directory, Response

from user_study.metadata import metadata_loader
from user_study.recommender import recommender

import logging

logger = logging.getLogger(__name__)

logging.basicConfig(
    #filename='music_2gram.out',
    #filemode='a',
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

app = Flask(__name__)



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


@app.route('/user_review', methods=['POST'])
def user_review():
    data = request.form.to_dict()
    print(data)
    with open("user_study/user_study_reviews.json", "a") as f:
        f.write(json.dumps(data) + '\n')
    user_id = data['user_id']
    recommender.generate_recommendations_async(user_id)
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
    return recommender.get_recommendations_of(user_id)


@app.route('/submit_user_evaluation', methods=['POST'])
def submit_user_evaluation():
    # save score
    pass


# main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    app.run()
