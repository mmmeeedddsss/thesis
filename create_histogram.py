import numpy as np
import matplotlib.pyplot as plt

P_lower = 98
P_upper = 97

"""
with open('idf_values.txt', 'r') as f:
    s = f.read()
    idfs = eval(s)
    values = [v for k, v in idfs.items()]

"""
with open('idf_values_test.txt', 'r') as f:
    s = f.read()
    idfs = eval(s)
    values = [v for k, v in idfs.items()]

unbiased_freq_dict = {}

import csv
with open('dataset/unigram_freq.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file, )
    for row in csv_reader:
        unbiased_freq_dict[row['word']] = int(row['freq'])


unbiased_freqs = [v for k, v in unbiased_freq_dict.items()]

upper_unbiased_freqs = np.percentile(unbiased_freqs, P_upper)

print(upper_unbiased_freqs)


lower = np.percentile(values, P_lower)

c = 0

numpy_arr = np.array(unbiased_freqs)

for k, v in idfs.items():
    if v >= lower:
        if k not in unbiased_freq_dict or unbiased_freq_dict[k] > upper_unbiased_freqs:
            continue
        print(k)
        c += 1

print(lower, np.average(values))
print(f'Have {c} words')


exit(0)

bins = 500
hist, bin_edges = np.histogram(values, bins=bins)


plt.plot(hist)
plt.axvline(x=upper, color='r')
plt.axvline(x=lower, color='r')
plt.ylabel('some numbers')
plt.show()
