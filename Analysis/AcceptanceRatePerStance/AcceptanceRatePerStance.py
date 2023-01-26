from matplotlib import pyplot as plt
from matplotlib import ticker as mtick
import seaborn as sns
import numpy as np
import pandas as pd
import json
from scipy.ndimage import gaussian_filter1d

sns.set_theme(style='whitegrid')

data = pd.read_json(r'../../Data/Model Predicted Data/AcceptedRejected-Predictions.json').to_dict(orient='records')

cp = sns.color_palette()[:8] + sns.color_palette('muted')[:8]

# calculate overall acceptance rate
overall_rate = sum(1 for p in data if p['status'] == 'accepted') / len(data)
print('Overall acceptance rate: %.4f' % overall_rate)

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

# create plot
stats = df.groupby(['stance'])['accepted'].agg(['mean', 'count', 'std'])
ci95hi, ci95lo = list(), list()
for i in stats.index:
	m, c, s = stats.loc[i]
	ci95hi.append(m + 1.95*s/np.sqrt(c))
	ci95lo.append(m - 1.95*s/np.sqrt(c))
stats['ci95hi'] = ci95hi
stats['ci95lo'] = ci95lo
for c in stats.columns: stats[c] = gaussian_filter1d(stats[c], sigma=1)

stats['ci95lo'] = np.clip(stats['ci95lo'], 0, 1)

plt.figure(figsize=(5.4, 4.0))
plt.hlines(overall_rate, -1, 1, linestyles='dotted', color='black')
plt.plot(stats['mean'], color='black')
plt.fill_between(stats.index, stats['ci95lo'], stats['ci95hi'], edgecolor='black', facecolor='black', alpha=0.06)

plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))

plt.title('Acceptance Rate per Stance')
plt.xlabel('Stance')
plt.ylabel('Acceptance Rate')
plt.subplots_adjust(bottom=0.14, left=0.15, right=0.98)
plt.savefig('AcceptanceRatePerStance.pdf')
plt.show()
