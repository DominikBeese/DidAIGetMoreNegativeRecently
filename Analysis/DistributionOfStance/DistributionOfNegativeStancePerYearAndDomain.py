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


# negative stance per year and domain (Smooth Lineplot)
plt.figure(figsize=(5.4, 4.0))
plt.subplots_adjust(right=0.98, bottom=0.13)
for k, df in enumerate(dfs):
	# binarize stance
	dft = df.copy(deep=True)
	dft['stance'] = np.where(dft['stance'] <= -0.1, 'Negative', 'Not Negative')
	
	# percentages per year
	dft = dft.groupby(['stance', 'year']).size().reset_index().pivot(columns='stance', index='year', values=0).fillna(0)
	dft = dft.div(dft.sum(1), axis=0)
	dft['Negative'] = gaussian_filter1d(dft['Negative'], sigma=2)
	
	# plot
	plt.plot(dft['Negative'], color=cp[k*3], label=['NLP', 'ML'][k])

plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
plt.title('Distribution of Negative Stance per Year and Domain')
plt.xlabel('Year')
plt.ylabel('Percentage of Papers per Year')
plt.ylim((0, None))
plt.legend()
plt.savefig('DistributionOfNegativeStancePerYearAndDomain.pdf')
plt.show()
