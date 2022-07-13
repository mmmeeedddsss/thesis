import sys
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

num_bins = 100

fname = sys.argv[1]

bins = []
for i in range(num_bins):
    bins.append({True: 0, False: 0})

with open(fname, 'r') as f:
    lines = f.readlines()
    for line in lines:
        dist, real = line.split(' ')
        dist = float(dist)
        real = int(real)
        # recommended = True if recommended == 'True' else False

        slot = int(dist * num_bins)
        bins[slot][real >= 4] += 1
x = []
y = []
nums = []

num_pos = 0
num_neg = 0

for i, bin in enumerate(bins):
    num_pos += bin[True]
    num_neg += bin[False]
    rat = bin[True] / (bin[True] + bin[False]) if bin[True] + bin[False] else 0
    # print(i, rat)
    if rat != 0:
        x.append(i / 100)
        y.append(rat)
        nums.append(bin[True] + bin[False])

plt.scatter(x, y, label='positive votes / all votes\n per distance')
plt.plot(x, y)
plt.axline((0, num_pos / (num_pos + num_neg)), (1, num_pos / (num_pos + num_neg)),
           linestyle='--', color='gray', label='avg ratio of\npositive votes / all votes')
plt.legend(loc='best')
print(num_pos / (num_pos + num_neg))

plt.title(f'Strategy - {fname}')

# naming the x axis
plt.xlabel('Match Score')
# naming the y axis
plt.ylabel('Ratio of like votes over all votes')

# function to show the plot
plt.savefig(f'{fname}.png', bbox_inches='tight')
plt.clf()

plt.scatter(x, nums)
plt.plot(x, nums)

plt.title(f'{fname} vote distribution by distance')

# naming the x axis
plt.xlabel('Match Score')
# naming the y axis
plt.ylabel('Ratio of like votes over all votes')

plt.savefig(f'{fname}_distribution.png', bbox_inches='tight')
plt.clf()

p, _ = pearsonr(x, y)
s, _ = spearmanr(x, y)

print(f'For strategy {fname}')
print('Pearson', p)
print('Spearman', s)
