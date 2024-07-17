from descriptive_nick import block_stats
from wrangler_nick import subjectDataWrangler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_loc = r'/Users/nick/Projects/ODoherty/vince_data/csv' # For mac
#data_loc = 'C:/Users/nrive/Projects/banditTask/vince_data/csv' # For pc
exclude = [0,1,2,3,4,5,14,24,34,37]

samSub = 8
samSubjectData = subjectDataWrangler(samSub, data_loc).subjectData
# RL/HM Split
block_score_agg = np.zeros((len(samSubjectData['block']), 97))+np.nan
isRL1HM2 = np.zeros((97))

for sub in range(97): 
    if(sub not in exclude):     
        # Data after 'wrangling'
        subjectData = subjectDataWrangler(sub, data_loc).subjectData
        block = block_stats(subjectData)
        block_score_agg[:len(block.block_score), sub] = np.nanmean(block.block_score, axis=1)

        if(subjectData['instructCond'][0]=='rl'):
            isRL1HM2[sub] = 1
        elif(subjectData['instructCond'][0]=='hm'):
            isRL1HM2[sub] = 2


fig, ax = plt.subplots(3,1, sharex=True)
ax[0].plot(block_score_agg)
ax[0].set_title('Raw Wins')
ax[1].plot(np.nanmean(block_score_agg, axis=1))
ax[1].set_title('Aggregated Wins per Trial Length')
counts = np.sum(~np.isnan(block_score_agg), axis=1)
ax[2].plot(counts)
ax[2].set_title('Subject Counts')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(2,1, figsize=(20,8))
ax[0].plot(np.nanmean(block_score_agg[:,(isRL1HM2==1)], axis=1), label='RL')
ax[0].plot(np.nanmean(block_score_agg[:,(isRL1HM2==2)], axis=1), label='HM')
ax[0].legend()
ax[0].set_title('p(Win)')
ax[1].plot(np.sum(~np.isnan(block_score_agg[:,(isRL1HM2==1)]), axis=1), label='RL')
ax[1].plot(np.sum(~np.isnan(block_score_agg[:,(isRL1HM2==2)]), axis=1), label='HM')
ax[1].legend()
ax[1].set_title('Subject Counts by Trial Length')
plt.show()