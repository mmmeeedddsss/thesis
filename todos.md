
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
    2. Transfer learning on pretrained word2vec (did a different thing: used pretrained and go with custom as fallback strategy)
2. Try tf-idf to eliminate common words like good song 
    1. Try tf-idf with reducing the weights of words (Used idf values only, worked fine, used tfidf directly for keywords, bert and yake is better)
        - Multiplying the distance coming from w2v with w1 * w2 where w's are 1/idf values
        - Pipeline dramatically slowed down
    2. Aggregate words using keyword extraction weights (not done)
3. KL Divergence 


18 Aug 2021 Meeting notes

----- Start to take notes -----

Decided Action Items:

- Entropy 
    - Can we benefit from calculating some entropy for each user. 
        Given rating, what is the probably of using the word w on review
        - Entropy, conditional entropy
    - we can both filter the users below a threshold(percentage) or weight them with entropy values
    2. Aggregate words using keyword extraction weights (not done)
    3. Try movie db, maybe use neo4j
KL Divergence - apply to user - item similarity


Next meeting: 8 Sept 2021 (15 Sept is actual)



----- Start to take notes -----

Decided Action Items:

- Formally write your pipeline, share the doc with handeo@gmail.com
- Try movie db, maybe use neo4j try personal pc


Next meeting: 6(13) Oct 2021


----- Start to take notes -----

Decided Action Items:

- ngram
- reference



Next meeting: 24 Nov 2021


----- Start to take notes -----

- try new algorithms for the word vectorizer step
 - sent2vec 
 - doc2vec 
 - skipThoughtVectors https://arxiv.org/abs/1506.0672
- movie database



Next meeting: 22 Dec 2021 - 12.00

----- Start to take notes -----

- word2vec with averaging set of two words
- write your thesis



Next meeting: 12 Jan 2022 - 12.00

----- Start to take notes -----

- check why 2 word features cannot be recommended
- start using same words on explanations
- literature search/survey


Next meeting: 26 Jan 2022 - 12.00

----- Start to take notes -----

- review how other do user studies, and do a small user study for testing
    - give a multiple choice of recommendations and ask what you would recommend to a person
- try to continue literature review part


Next meeting: ?? March 2022 - 12.00

- make bert versus yake on user study with the same commands
- change the ratings to sliders, look for likert scale



Next meeting: 30 March 2022 - 12.00

- use cds and vinyl dataset
