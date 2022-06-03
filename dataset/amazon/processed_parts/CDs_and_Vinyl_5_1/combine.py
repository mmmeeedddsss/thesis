import glob
import pandas as pd
from tqdm import tqdm

dfs = []
for file in tqdm(glob.glob("*.gzip")):
    dfs.append(pd.read_pickle(file))
    dfs[-1].drop(['style', 'vote', 'image'], axis=1, inplace=True)

df = pd.concat(dfs, ignore_index=True)
print(df)
df.to_pickle('CDs_and_Vinyl_1gram_combined_05.gzip')
