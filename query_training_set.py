import os
from dataset.amazon.loader import AmazonDatasetLoader
from models.nlp.KeyBERT import KeyBERTExtractor
from models.nlp.yake import YakeExtractor


amazon_dataloader = AmazonDatasetLoader()

df_all = amazon_dataloader.get_pandas_df()

filename = AmazonDatasetLoader.filenames[0].split('/')[-1].split('.')[0]





