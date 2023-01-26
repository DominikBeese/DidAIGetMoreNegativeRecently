from matplotlib import pyplot as plt
from matplotlib import ticker as mtick
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d

sns.set_theme(style='whitegrid')

dfs = [pd.read_json(r'../../Data/Model Predicted Data/%s-Predictions.json' % ds) for ds in ['NLP', 'ML']]
dfs = [df[['url', 'year', 'stance']].drop_duplicates() for df in dfs]

cp = sns.color_palette()[:8] + sns.color_palette('muted')[:8]


# average stance per year and domain (Smooth Lineplot)
plt.figure(figsize=(5.4, 4.0))
for k, df in enumerate(dfs):
	stats = df.groupby(['year'])['stance'].agg(['mean', 'count', 'std'])
	ci95hi, ci95lo = list(), list()
	for i in stats.index:
		m, c, s = stats.loc[i]
		ci95hi.append(m + 1.95*s/np.sqrt(c))
		ci95lo.append(m - 1.95*s/np.sqrt(c))
	stats['ci95hi'] = ci95hi
	stats['ci95lo'] = ci95lo
	for c in stats.columns: stats[c] = gaussian_filter1d(stats[c], sigma=1)
	
	plt.plot(stats['mean'], color=cp[k*3], label=['NLP', 'ML'][k])
	plt.fill_between(stats.index, stats['ci95lo'], stats['ci95hi'], edgecolor=cp[k*3], facecolor=cp[k*3], alpha=0.1)

plt.title('Average Stance per Year and Domain')
plt.xlabel('Year')
plt.ylabel('Average Stance')
plt.ylim(None, 1)
plt.legend(loc='upper left')
plt.subplots_adjust(bottom=0.13, right=0.98)
plt.savefig('AverageStancePerYearAndDomain.pdf')
plt.show()
