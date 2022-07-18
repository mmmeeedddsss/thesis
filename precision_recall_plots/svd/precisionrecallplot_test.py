import matplotlib.pyplot as plt







import matplotlib.transforms as mtransforms

#fig, axs = plt.subplots(3)


fig, axs = plt.subplot_mosaic([['balanced'], ['random'], ['tholds']],
                              constrained_layout=True)

fig.suptitle('Presicion-Recall Curve For exp/mean_of_all')
trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)


f1 = []

for i in range(len(recall_balanced)):
    f1.append(2*precision_balanced[i]*recall_balanced[i]/(precision_balanced[i]+recall_balanced[i])) \
        if precision_balanced[i]+recall_balanced[i] != 0 else f1.append(0)
axs['balanced'].plot(recall_balanced, precision_balanced)
axs['balanced'].set_xlabel('Recall')
axs['balanced'].set_ylabel('Precision')
axs['balanced'].axline((0, 0.4), (1, 0.4),
           linestyle='--', color='gray', label='expected')
trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
axs['balanced'].text(0.0, 1.0, 'balanced', transform=axs['balanced'].transAxes + trans,
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))


f1 = []

for i in range(len(recall)):
    f1.append(2*precision[i]*recall[i]/(precision[i]+recall[i])) if precision[i]+recall[i] != 0 else f1.append(0)
axs['random'].plot(recall, precision)
axs['random'].axline((0, 12914/(15000)), (1, 12914/(15000)),
           linestyle='--', color='gray', label='expected')
axs['random'].set_xlabel('Recall')
axs['random'].set_ylabel('Precision')
trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
axs['random'].text(0.0, 1.0, 'random', transform=axs['random'].transAxes + trans,
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))


tholds = [tholds[0]] + tholds
tholds_balanced = [tholds_balanced[0]] + tholds_balanced

axs['tholds'].plot(recall, tholds)
axs['tholds'].plot(recall_balanced, tholds_balanced)
axs['tholds'].set_xlabel('Recall')
axs['tholds'].set_ylabel('Tresholds')
trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
axs['tholds'].text(0.0, 1.0, 'thresholds', transform=axs['tholds'].transAxes + trans,
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))

axs['balanced'].axvspan(0.15, 0.20, color='#ffd152', alpha=0.3)
axs['random'].axvspan(0.15, 0.20, color='#ffd152', alpha=0.3)
axs['tholds'].axvspan(0.15, 0.20, color='#ffd152', alpha=0.3)


plt.show()