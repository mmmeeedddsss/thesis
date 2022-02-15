## Description 

Mert Tunc Thesis work


## Usage - Plain development

Yelp dataset: https://www.yelp.com/dataset/documentation/main
Amazon dataset: https://nijianmo.github.io/amazon/index.html

Following command runs and reports the recommendation metrics and some sample explanations to the test set.
```
python driver.py
```

## Usage - Get recommendations for yourself

Run the following for opening a flask server on your local 
```
python -m user_study.backend
```

Navigate to 
```
http://127.0.0.1:5000/
```
and start writing reviews. With each review written, an async task will be triggered 
for creating recommendations based on the reviews you shared.

After filling 5+ reviews, you can visit
```
http://127.0.0.1:5000/recommend
```
to see what is reviewed to you and why.
You can provide feedback here to contribute to user study here.
