from matplotlib import pyplot as plt
from matplotlib import ticker as mtick
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_theme(style='whitegrid')

df = pd.concat([pd.read_json(r'../../Data/Model Predicted Data/%s-Predictions.json' % ds) for ds in ['NLP', 'ML']])
df = df[['url', 'stance']].drop_duplicates()

cp = sns.color_palette()[:8] + sns.color_palette('muted')[:8]


# distribution of stance (Histplot and Bar)
fig, (ax1, ax2) = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [8, 1], 'wspace': 0.05}, figsize=(5.4, 4.0))

dft = df.copy(deep=True).reset_index()
sns.histplot(data=dft, x='stance', element='poly', fill=False, stat='probability', binrange=(-1.05, 1.05), bins=20, color='black', ax=ax1)
ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: str('%f'%(100*x)).rstrip('0').rstrip('.')+'%'))

dft = df.copy(deep=True)
dft['stance'] = np.where(dft['stance'] < 0, 'Negative', 'Positive')
dft['venue'] = 'Class'
dft = dft.groupby(['stance', 'venue']).size().reset_index().pivot(columns='stance', index='venue', values=0)
dft = dft[['Negative', 'Positive']]
dft = dft.div(dft.sum(1), axis=0).mul(100)
dft.plot(kind='bar', stacked=True, rot=0, color=[cp[3], cp[0]], legend=False, ax=ax2)

ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
ax2.tick_params(labelleft=False, labelright=True)

plt.subplots_adjust(top=0.93)
plt.suptitle('Distribution of Stance', fontsize='medium')
ax1.set_xlabel('Stance')
ax2.set_xlabel(None)
ax1.set_ylabel('Percentage of Papers')
ax2.set_ylabel(None)
plt.subplots_adjust(bottom=0.14, left=0.15, right=0.89)
plt.savefig('DistributionOfStance.pdf')
plt.show()
