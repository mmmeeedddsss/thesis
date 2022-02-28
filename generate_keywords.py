import os

import pandas as pd
import numpy as np
from dataset.amazon.loader import AmazonDatasetLoader
from dataset.yelp.loader import YelpDatasetLoader
from models.nlp.KeyBERT import KeyBERTExtractor
from models.nlp.yake import YakeExtractor

output_dir = 'processed_df_cache'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

amazon_dataloader = AmazonDatasetLoader()
#yelp_dataloader = YelpDatasetLoader()

df_all = amazon_dataloader.get_pandas_df()

filename = AmazonDatasetLoader.filenames[0].split('/')[-1].split('.')[0]

df_pieces = np.array_split(df_all, 20)

for i, df in enumerate(df_pieces):
    current_part_path = f'{output_dir}/{filename}_{i}.gzip'
    if os.path.exists(current_part_path):
        print(f'-------------- Skipping {i} --------------')
        continue

    print(f'-------------- Starting {i} --------------')

    df = KeyBERTExtractor().extract_keywords(df, {'top_n': 10, 'keyphrase_ngram_range': (1, 2)})

    df = YakeExtractor().extract_keywords(df)
    df.to_pickle(current_part_path)

    """
    from models.nlp.tfidf import tfidfExtractor
    df = tfidfExtractor().extract_keywords(df)
    df.to_pickle('Digital_Music_5_with_extracted_topics.gzip')
    """


