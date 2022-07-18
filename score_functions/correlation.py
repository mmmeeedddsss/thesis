import sys
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import matplotlib.transforms as mtransforms

num_bins = 100

fname_balanced = sys.argv[1]
if 'balanced' in fname_balanced:
    fname_random = fname_balanced.replace('balanced', 'random')
else:
    fname_random = fname_balanced
    fname_balanced = fname_balanced.replace('random', 'balanced')

fnames = [(fname_random, 'random'), (fname_balanced, 'balanced')]



fig, axs = plt.subplot_mosaic([['balanced', 'balanced_dist'], ['random', 'random_dist']],
                              constrained_layout=True)

s_func = fname_balanced.split('/')[2].replace('_', ' ')
kw_extractor = fname_balanced.split('/')[1].replace('_', ' ')



fig.suptitle(f'Scoring function: {s_func}, Keyword Extractor: {kw_extractor}')
trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)

for fname, exp_type in fnames:
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



    axs[exp_type].plot(x, y)
    axs[exp_type].scatter(x, y, s=10)
    axs[exp_type].set_xlabel('Match Score')
    axs[exp_type].set_ylabel('Rw(Positive)/Rw(All)')
    axs[exp_type].axline((0, num_pos/(num_pos+num_neg)), (1, num_pos/(num_pos+num_neg)),
               linestyle='--', color='gray', label='expected')
    trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
    axs[exp_type].text(0.0, 1.0, exp_type, transform=axs[exp_type].transAxes + trans,
                fontsize='medium', verticalalignment='top', fontfamily='serif',
                bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))

    exp_type_dist = exp_type + '_dist'


    axs[exp_type_dist].plot(x, nums)
    axs[exp_type_dist].scatter(x, nums, s=10)
    axs[exp_type_dist].set_xlabel('Match Score')
    axs[exp_type_dist].set_ylabel('Rw(Positive)/Rw(All)')
    #axs['balanced'].axline((0, 0.4), (1, 0.4),
    #           linestyle='--', color='gray', label='expected')
    trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
    axs[exp_type_dist].text(0.0, 1.0, exp_type + ' - data histogram', transform=axs[exp_type_dist].transAxes + trans,
                fontsize='medium', verticalalignment='top', fontfamily='serif',
                bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))


    p, _ = pearsonr(x, y)
    s, _ = spearmanr(x, y)

    print(f'For strategy {fname}')
    print('Pearson', p)
    print('Spearman', s)

fig.set_size_inches(8, 7, forward=True)

plt.savefig(f'{fname_balanced.replace("_balanced", "")}_figures.png', bbox_inches='tight')

plt.savefig(f'sf_{kw_extractor.lower()}_{s_func}_figures.png', bbox_inches='tight')