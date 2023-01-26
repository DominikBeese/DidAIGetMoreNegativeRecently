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


def replaceNaN(s):
    r, p = dict(), None
    for x in s.index:
        if not np.isnan(s[x]):
            r[x] = s[x]
            p = r[x]
        else:
            if p is None: p = s[x+1]
            r[x] = p
    return pd.Series(r)

# average positive and negative stance per year and domain (Smooth Lineplot Duo)
plt.figure(figsize=(10.0, 4.0))
plt.subplots_adjust(left=0.06, right=0.99, bottom=0.14, wspace=0.25, top=0.90)
for r, fs in enumerate([1, -1]):
	plt.subplot(1, 2, r+1)
	for k, df in enumerate(dfs):
		dft = df.copy(deep=True)
		dft = dft[dft['stance'] * fs >= 0.1] # positive/negative only
		stats = dft.groupby(['year'])['stance'].agg(['mean', 'count', 'std'])
		ci95hi, ci95lo = list(), list()
		for i in stats.index:
			m, c, s = stats.loc[i]
			ci95hi.append(m + 1.95*s/np.sqrt(c))
			ci95lo.append(m - 1.95*s/np.sqrt(c))
		stats['ci95hi'] = ci95hi
		stats['ci95lo'] = ci95lo
		for c in stats.columns: stats[c] = gaussian_filter1d(replaceNaN(stats[c]), sigma=1 if fs == 1 else 2)
		
		plt.plot(stats['mean'], color=cp[k*3], label=['NLP', 'ML'][k])
		plt.fill_between(stats.index, stats['ci95lo'], stats['ci95hi'], edgecolor=cp[k*3], facecolor=cp[k*3], alpha=0.1)
		
		s = {1: 'Positive', -1: 'Negative'}[fs]
		plt.title('Average %s Stance per Year and Domain' % s)
		plt.xlabel('Year')
		plt.ylabel('Average %s Stance' % s)
		plt.ylim(-1 if fs == -1 else None, 1 if fs == 1 else None)
		plt.legend(loc='lower right')

plt.savefig('AveragePositiveNegativeStancePerYearAndDomain.pdf')
plt.show()
