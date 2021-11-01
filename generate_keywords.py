
import pandas as pd
import numpy as np
from dataset.amazon.loader import AmazonDatasetLoader
from dataset.yelp.loader import YelpDatasetLoader
from models.nlp.KeyBERT import KeyBERTExtractor
from models.nlp.yake import YakeExtractor


amazon_dataloader = AmazonDatasetLoader()
#yelp_dataloader = YelpDatasetLoader()

df = amazon_dataloader.get_pandas_df()




#%%

df = KeyBERTExtractor().extract_keywords(df, {'top_n': 15, 'keyphrase_ngram_range': (1, 2)})

#%%

df.to_pickle('Digital_Music_5_with_extracted_topics.gzip')
df = YakeExtractor().extract_keywords(df)



#%%

#reviews = df['review'].to_list()
#res = KeyBERTExtractor().model.extract_keywords(reviews)


#%%

from models.nlp.tfidf import tfidfExtractor


#df = amazon_dataloader.get_processed_pandas_df()

df = tfidfExtractor().extract_keywords(df)

#%%

df.to_pickle('Digital_Music_5_with_extracted_topics.gzip')

#%%



#%%




