from keybert import KeyBERT
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import yake

abstract = """
Explanation in the recommendations is an important aspect to have in many kinds of marketplace applications to share reasoning and context with the users in addition to the recommended item. In this thesis, an innovative method for generating explainable recommendations is designed, implemented and tested. 
The proposed design consists of extracting some phrases from the user written review texts, assigning these phrases to the users as preferences and to items as their features, and then generating recommendations using the similarities between these assigned phrases. In such a design, since the recommendations are made using human understandable phrases, the same phrases can be used to explain the reasoning of the recommendations.
There was not much study that uses keyword extraction techniques and word vectorizers to generate recommendations. Due to the lack of work in the area, it is decided to study such an algorithm that use these techniques. The expectation was to have a recommender that performs worse than the state of the art Deep Learning models but still, we were expecting to have a recommender that can generate sane results and additionally have the ability to explain its reasoning. 
To evaluate the design, alongside of calculating numerical results for the quality of the recommender, a user study with <fill> people is conducted. These experiments shown that <x> percent of the recommendations are liked by the people while <y> percent of the explanations for the recommended items are found meaningful.
"""

kw_extractor = yake.KeywordExtractor(n=1, top=8)
keywords = kw_extractor.extract_keywords(abstract)

print('yake1')
for kw in keywords:
    print(kw[0]+', ', end='')

kw_extractor = yake.KeywordExtractor(n=2, top=8)
keywords = kw_extractor.extract_keywords(abstract)


print('yake2')
for kw in keywords:
    print(kw[0]+', ', end='')

exit(1)

kw_model = KeyBERT()
keywords = kw_model.extract_keywords(abstract, keyphrase_ngram_range=(1,1), top_n=8)

print('keybert1:', keywords)

keywords = kw_model.extract_keywords(abstract, keyphrase_ngram_range=(1,2), top_n=8)

print('keybert2:', keywords)



binary1 = np.array([[1395, 825], [105, 175]])

fig, ax = plot_confusion_matrix(conf_mat=binary1, colorbar=True)
fig.suptitle('Confusion matrix for the predictions on balanced sampled dataset')
plt.show()


binary1 = np.array([[1890, 10327],[157, 2626]])

fig, ax = plot_confusion_matrix(conf_mat=binary1, colorbar=True)
fig.suptitle('Confusion matrix for the predictions on random sampled dataset')
plt.show()


exit(1)