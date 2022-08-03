import json
from math import floor

import pandas as pd
import numpy as np
from pandasql import sqldf
import matplotlib.pyplot as plt

sql = lambda q: sqldf(q, globals())

def d_histogram(ls, label, color=None):
    ls = [int(x) for x in ls]
    plt.clf()
    if color:
        plt.hist(ls, bins=10, range=(0, 100), color="#f5a214")
    else:
        plt.hist(ls, bins=10, range=(0, 100))
    plt.title(label)
    plt.savefig(f"{label.replace(' ', '_')}.png", bbox_inches='tight')


with open("ratings.json", "r") as f:
    ratings = json.load(f)

df = pd.DataFrame(ratings).replace({'bert_1': 'Bert 1', 'yake_1': 'Yake 1', 'yake': 'Yake 2'})

print(df)

df = sql("select * from df where recommender != 'bert' order by recommender, user_id")

print()
print("Num of users")
print(sql("SELECT count(distinct (user_id)) FROM df "))

print()
print("Num of rows per recommender ")
row_per_reco = sql("SELECT recommender, count(*) as number_of_ratings FROM df group by recommender order by recommender")
print(row_per_reco)
row_per_reco = row_per_reco.set_index('recommender')
ax = row_per_reco.plot(kind='pie', y='number_of_ratings',
                  autopct=lambda x: str(floor(x*(row_per_reco.sum()/100))))
frame1 = plt.gca()
frame1.axes.get_yaxis().set_visible(False)
frame1.axes.get_xaxis().set_visible(False)
h, l = ax.get_legend_handles_labels()
ax.get_legend().remove()
#ax.legend(h[:3], ["Bert 1", "Yake", "Yake 1"], loc=3, fontsize=12)
plt.title("Number of rated items per recommender")

plt.savefig(f'num_rated_items.png', bbox_inches='tight')
#plt.show()

print()
print("Num of users per recommender ")
user_per_recommender = sql("SELECT recommender, count(distinct(user_id)) as user_per_reco FROM df group by recommender").set_index('recommender')
print(user_per_recommender)
ax = user_per_recommender.plot(kind='pie', y='user_per_reco',
                               autopct=lambda x: round(((x*user_per_recommender.sum())/100).item()))
frame1 = plt.gca()
frame1.axes.get_yaxis().set_visible(False)
frame1.axes.get_xaxis().set_visible(False)
h, l = ax.get_legend_handles_labels()
ax.get_legend().remove()
#ax.legend(h[:3], ["Bert 1", "Yake", "Yake 1"], loc=3, fontsize=12)
plt.title("Number of users attended to user study")

plt.savefig(f'num_users_per_reco.png', bbox_inches='tight')

print()
print("Num of average ratings per user per recommender")
per_user_recommender = sql("SELECT recommender, user_id, avg(recommendation_rating) as avg_reco_rate, avg(explanation_rating) as avg_exp_rate FROM df group by recommender, user_id order by recommender, user_id")
print(per_user_recommender)

print()
print("Num high ratings")
num_high_ratings = sql("SELECT count(*) FROM df where explanation_rating>=65 group by recommender order by recommender")
print(num_high_ratings)

print()
print("Num all ratings")
num_high_ratings = sql("SELECT count(*) FROM df group by recommender order by recommender")
print(num_high_ratings)

print()
print("Ratings per user per recommender")
per_recommender_individual = sql("SELECT recommender, recommendation_rating FROM df order by recommender")

per_recommender_individual = per_recommender_individual.set_index('recommender')

print(per_recommender_individual)

bert_1 = sql("SELECT recommendation_rating FROM per_recommender_individual where recommender = 'Bert 1'").values.flatten().tolist()
yake_1 = sql("SELECT recommendation_rating FROM per_recommender_individual where recommender = 'Yake 1'").values.flatten().tolist()
yake_2 = sql("SELECT recommendation_rating FROM per_recommender_individual where recommender = 'Yake 2'").values.flatten().tolist()

d_histogram(bert_1, 'Bert 1 - Rating distribution for the recommended items')
d_histogram(yake_1, 'Yake 1 - Rating distribution for the recommended items')
d_histogram(yake_2, 'Yake 2 - Rating distribution for the recommended items')


print()
print("Explanation ratings per user per recommender")
per_recommender_individual = sql("SELECT recommender, explanation_rating FROM df order by recommender")

per_recommender_individual = per_recommender_individual.set_index('recommender')

print(per_recommender_individual)

bert_1 = sql("SELECT explanation_rating FROM per_recommender_individual where recommender = 'Bert 1'").values.flatten().tolist()
yake_1 = sql("SELECT explanation_rating FROM per_recommender_individual where recommender = 'Yake 1'").values.flatten().tolist()
yake_2 = sql("SELECT explanation_rating FROM per_recommender_individual where recommender = 'Yake 2'").values.flatten().tolist()

d_histogram(bert_1, 'Bert 1 - Rating distribution for the explanations', color=True)
d_histogram(yake_1, 'Yake 1 - Rating distribution for the explanations', color=True)
d_histogram(yake_2, 'Yake 2 - Rating distribution for the explanations', color=True)



print()
print("Num of average rating per recommender")
per_recommender = sql("SELECT recommender, avg(recommendation_rating) as `Average rating given of recommendations [0-100]`, avg(explanation_rating) as `Average rating given of explanations [0-100]` FROM df group by recommender order by recommender")
print(per_recommender)
per_recommender = per_recommender.set_index('recommender')
per_recommender.plot(kind='bar')
plt.title("Mean of user rating averages to recommenders")
plt.xticks(rotation=45, ha='right')
plt.legend(loc="lower right")
plt.savefig(f'mean_recommender.png', bbox_inches='tight')

print()
df_t = sql("select * from per_user_recommender where recommender != 'bert'")
print(df_t)

plt.clf()


bert_1 = sql("select avg_reco_rate from df_t where recommender == 'Bert 1'").values.flatten().tolist()
yake_1 = sql("select avg_reco_rate from df_t where recommender == 'Yake 1'").values.flatten().tolist()
yake_2 = sql("select avg_reco_rate from df_t where recommender == 'Yake 2'").values.flatten().tolist()

bert_1_raw = sql("select avg(recommendation_rating) from df where recommender == 'Bert 1'").values.flatten().tolist()

print(len(bert_1))
print(len(yake_1))
print(len(yake_2))

bert_1.insert(7, np.mean(bert_1))
yake_1.insert(8, np.mean(yake_1))

d_histogram(bert_1, 'Bert 1 - Mean Rating of the recommended item per user')
d_histogram(yake_1, 'Yake 1 - Mean Rating of the recommended item per user')
d_histogram(yake_2, 'Yake 2 - Mean Rating of the recommended item per user')


from scipy import stats
print(stats.friedmanchisquare(bert_1, yake_1, yake_2))


bert_1 = sql("select avg_exp_rate from df_t where recommender == 'Bert 1'").values.flatten().tolist()
yake_1 = sql("select avg_exp_rate from df_t where recommender == 'Yake 1'").values.flatten().tolist()
yake_2 = sql("select avg_exp_rate from df_t where recommender == 'Yake 2'").values.flatten().tolist()

bert_1.insert(7, np.mean(bert_1))
yake_1.insert(8, np.mean(yake_1))

d_histogram(bert_1, 'Bert 1 - Mean Rating of the explanation per user', color=True)
d_histogram(yake_1, 'Yake 1 - Mean Rating of the explanation per user', color=True)
d_histogram(yake_2, 'Yake 2 - Mean Rating of the explanation per user', color=True)

# Q, p
print(stats.friedmanchisquare(bert_1, yake_1, yake_2))



