
21 May 2021 Meeting notes

Decided Action Items:
1. Classifier todos 
    1. Ordinal classification - Done
    2. mark missing numbers in X matrix - Not supported on sklearn impl of classifiers
    3. investigate if 0 is the best solution for unknown values - Imputation - DONE
2. SVD optimization - DONE
3. At least one more dataset
4. How explainable is it? - Partially Done
    - Explain most likely predictions case by case
    - Explain least likely predictions case by case
    
    
9 June 2021 Meeting notes

Decided Action Items:

1. Find a pretrained word2vec and try it
    1. Play with word2vec with common words and get most similars (seems like working?)
    2. Transfer learning on pretrained word2vec (did a different thing)
2. Try tf-idf to eliminate common words like good song 
    1. Try tf-idf with reducing the weights of words (Used idf values only, worked fine, used tfidf directly for keywords)
    2. Aggregate words using keyword extraction weights (not done)
3. KL Divergence 



Next meeting: 30 of June 12.00
Delayed date: 14 of July 12.00
