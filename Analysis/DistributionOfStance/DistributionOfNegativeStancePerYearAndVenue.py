from matplotlib import pyplot as plt
from matplotlib import ticker as mtick
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d

sns.set_theme(style='whitegrid')

VENUES = ['ACL', 'EMNLP', 'COLING', 'NAACL', 'SemEval', 'CoNLL', 'CL', 'TACL', 'NeurIPS', 'AAAI', 'ICML', 'ICLR']
dfs = [pd.read_json(r'../../Data/Model Predicted Data/%s-Predictions.json' % ds) for ds in ['NLP', 'ML']]
dfs = [df[['year', 'venue', 'url', 'stance']] for df in dfs]

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

# negative stance per year and venue (Smooth Lineplot Grid)
df = pd.concat(dfs)
dft = df.copy(deep=True)
dft['stance'] = np.where(dft['stance'] <= -0.1, 'Negative', 'Not Negative')
dfl = list()
for venue in VENUES:
	h = dft[dft['venue'] == venue]
	h = h.groupby(['stance', 'year']).size().reset_index().pivot(columns='stance', index='year', values=0).fillna(0)
	h = h.div(h.sum(1), axis=0).mul(100)
	h['Negative'] = gaussian_filter1d(replaceNaN(h['Negative']), sigma=2)
	dfl.append(h['Negative'])
dft = pd.DataFrame(dict(zip(VENUES, dfl)))
dft = dft.unstack().reset_index()
dft = dft.rename(columns={'level_0': 'venue', 0: 'percentage'})

fig, (a, b, c) = plt.subplots(figsize=(5.4, 4.0), ncols=4, nrows=3)
axes = [*a, *b, *c]

for i, venue in enumerate(VENUES):
	ax = axes[i]
	show_xlabel = i>=4*2
	show_ylabel = i%4==0
	
	sns.lineplot(data=dft[dft['venue'] == venue], x='year', y='percentage', color=cp[i], ax=ax)
	
	ax.set_xlim((1982.15, 2022.85))
	ax.set_ylim((0.3, 20))
	
	ax.set_xticks([2000, 2020])
	ax.set_xticklabels(['2000', '2020    '])
	ax.set_yscale('log')
	ax.set_yticks([1, 10])
	ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: str(int(x))+'%'))
	ax.tick_params(labelsize=10)
	ax.set_title(venue, fontsize=10.5, pad=2)
	
	ax.set_ylabel(None)
	if show_xlabel: ax.set_xlabel('Year')
	else:
		ax.set_xlabel(None)
		ax.tick_params(labelbottom=False)
	if show_ylabel: ax.set_ylabel('Papers')
	else:
		ax.set_ylabel(None)
		ax.tick_params(labelleft=False)

plt.subplots_adjust(left=0.12, bottom=0.12, right=0.98, top=0.885, wspace=0.08, hspace=0.30)
plt.suptitle('Distribution of Negative Stance per Year and Venue', fontsize='medium')
plt.savefig('DistributionOfNegativeStancePerYearAndVenue.pdf')
plt.show()
