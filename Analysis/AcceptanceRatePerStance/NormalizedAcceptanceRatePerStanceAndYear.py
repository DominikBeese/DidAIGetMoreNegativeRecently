from matplotlib import pyplot as plt
from matplotlib import ticker as mtick
import seaborn as sns
import numpy as np
import pandas as pd
import json
from scipy.ndimage import gaussian_filter1d

sns.set_theme(style='whitegrid')

data = pd.read_json(r'../../Data/Model Predicted Data/AcceptedRejected-Predictions.json').to_dict(orient='records')

YEARS = {
	'2015-2021': [2015, 2016, 2017, 2018, 2019, 2020, 2021],
	'2007-2014': [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014],
}

cp = sns.color_palette()[:8] + sns.color_palette('muted')[:8]

# calculate overall acceptance rate
overall_rate = sum(1 for p in data if p['status'] == 'accepted') / len(data)

# create dataframe for sections of stance
min_x, max_x, N_x = -1, 1, 2*7+1 ## modify 15/30 to change resolution
w_x = (max_x - min_x) / (N_x - 1)
df = pd.DataFrame([
	{
	'stance': next(x for x in np.linspace(min_x, max_x, N_x) if p['stance'] >= x-w_x/2 and p['stance'] < x+w_x/2),
	'accepted': {'accepted': 1, 'rejected': 0}[p['status']],
	'year': p['year'],
	}
	for p in data
])

def replaceNaN(s):
    r, p = dict(), None
    for x in s.index:
        if not np.isnan(s[x]):
            r[x] = s[x]
            p = r[x]
        else: r[x] = p
    return pd.Series(r)
	
# create plot
plt.figure(figsize=(5.4, 4.0))
for k, (year, years) in enumerate(YEARS.items()):
	# filter years
	dft = df.copy(deep=True)
	dft = dft[dft['year'].isin(years)]
	
	# normalize
	stats = dft['accepted'].agg(['mean', 'std'])
	dft['normalized_rate'] = (dft['accepted'] - stats['mean']) # / stats['std']
	
	# calculate stats
	stats = dft.groupby(['stance'])['normalized_rate'].agg(['mean', 'count', 'std'])
	ci95hi, ci95lo = list(), list()
	for i in stats.index:
		m, c, s = stats.loc[i]
		ci95hi.append(m + 1.95*s/np.sqrt(c))
		ci95lo.append(m - 1.95*s/np.sqrt(c))
	stats['ci95hi'] = ci95hi
	stats['ci95lo'] = ci95lo
	for c in stats.columns: stats[c] = gaussian_filter1d(replaceNaN(stats[c]), sigma=2)
	
	plt.plot(stats['mean'], color=cp[k], label=year)
	plt.fill_between(stats.index, stats['ci95lo'], stats['ci95hi'], edgecolor=cp[k], facecolor=cp[k], alpha=0.1)

plt.hlines(0, -1, 1, linestyles='dotted', color='black')

plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))

plt.title('Normalized Acceptance Rate per Stance and Year')
plt.xlabel('Stance')
plt.ylabel('Normalized Acceptance Rate')
plt.legend()
plt.subplots_adjust(bottom=0.14, left=0.15, right=0.98)
plt.savefig('NormalizedAcceptanceRatePerStanceAndYear.pdf')
plt.show()
