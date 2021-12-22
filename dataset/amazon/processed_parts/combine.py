import glob
import pandas as pd
from tqdm import tqdm

dfs = []
for file in tqdm(glob.glob("*.gzip")):
    dfs.append(pd.read_pickle(file))
    dfs[-1].drop(['style', 'vote', 'image'], axis=1, inplace=True)

df = pd.concat(dfs[:3], ignore_index=True)
print(df)
df.to_pickle('Movies_and_TV_5_unified_02inc.gzip')
