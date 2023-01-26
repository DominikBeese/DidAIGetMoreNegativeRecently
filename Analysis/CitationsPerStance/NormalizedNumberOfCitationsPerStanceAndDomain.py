from matplotlib import pyplot as plt
from matplotlib import ticker as mtick
import seaborn as sns
import numpy as np
import pandas as pd
import json
from scipy.ndimage import gaussian_filter1d

sns.set_theme(style='whitegrid')

dfs = [pd.read_json(r'../../Data/Model Predicted Data/%s-Predictions.json' % ds) for ds in ['NLP', 'ML']]
dfs = [df[['url', 'year', 'stance', 'citations']].drop_duplicates() for df in dfs]

cp = sns.color_palette()[:8] + sns.color_palette('muted')[:8]

plt.figure(figsize=(5.4, 4.0))
for k, domain in enumerate(['NLP', 'ML']):
	# filter venues
	data_filtered = dfs[k].to_dict(orient='records')
	
	# create dataframe for sections of stance
	min_x, max_x, N_x = -1, 1, 2*10+1 ## modify 15/30 to change resolution
	w_x = (max_x - min_x) / (N_x - 1)
	df = pd.DataFrame([
		{
		'stance': next(x for x in np.linspace(min_x, max_x, N_x) if p['stance'] >= x-w_x/2 and p['stance'] < x+w_x/2),
		'citations': p['citations'],
		'year': p['year'],
		}
		for p in data_filtered
	])

	# calculate normalized citations
	stats = df.groupby(['year'])['citations'].agg(['mean', 'std'])
	df['normalized_citations'] = (df['citations'] - list(stats['mean'][df['year']])) / list(stats['std'][df['year']])
	
	# create plot
	stats = df.groupby(['stance'])['normalized_citations'].agg(['mean', 'count', 'std'])
	ci95hi, ci95lo = list(), list()
	for i in stats.index:
		m, c, s = stats.loc[i]
		ci95hi.append(m + 1.95*s/np.sqrt(c))
		ci95lo.append(m - 1.95*s/np.sqrt(c))
	stats['ci95hi'] = ci95hi
	stats['ci95lo'] = ci95lo
	for c in stats.columns: stats[c] = gaussian_filter1d(stats[c], sigma=2)

	plt.hlines(0, -1, 1, linestyles='dotted', color='black')
	plt.plot(stats['mean'], color=cp[k*3], label=domain)
	plt.fill_between(stats.index, stats['ci95lo'], stats['ci95hi'], edgecolor=cp[k*3], facecolor=cp[k*3], alpha=0.1)

plt.legend()
plt.title('Normalized Number of Citations per Stance and Domain')
plt.xlabel('Stance')
plt.ylabel('Normalized Number of Citations')
plt.subplots_adjust(bottom=0.14, left=0.15, right=0.98)
plt.savefig('NormalizedNumberOfCitationsPerStanceAndDomain.pdf')
plt.show()
