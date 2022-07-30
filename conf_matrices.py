from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

binary1 = np.array([[1395, 825], [105, 175]])

fig, ax = plot_confusion_matrix(conf_mat=binary1, colorbar=True)
fig.suptitle('Confusion matrix for the predictions on balanced sampled dataset')

plt.savefig(f'conf_mat_balanced_best.png', bbox_inches='tight')


binary1 = np.array([[1890, 10327],[157, 2626]])

fig, ax = plot_confusion_matrix(conf_mat=binary1, colorbar=True)
fig.suptitle('Confusion matrix for the predictions on random sampled dataset')

plt.savefig(f'conf_mat_balanced_random.png', bbox_inches='tight')
