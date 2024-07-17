from descriptive_nick import block_stats
from wrangler_nick import subjectDataWrangler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#data_loc = r'/Users/nick/Projects/ODoherty/vince_data/csv' # For mac
data_loc = 'C:/Users/nrive/Projects/banditTask/vince_data/csv' # For pc
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


fig, ax = plt.subplots(1,1, sharex=True, sharey=True)
plt.plot(np.nanmean(block_score_agg, axis=1))
counts = np.sum(~np.isnan(block_score_agg), axis=1)
plt.plot(counts)
plt.show()

fig, ax = plt.subplots(4,1, sharex=True, sharey=True)
ax[0].plot(block_score_agg[:,(isRL1HM2==1)])
ax[1].plot(block_score_agg[:,(isRL1HM2==2)])
ax[2].plot(np.nanmean(block_score_agg[:,(isRL1HM2==1)], axis=1), label='RL')
ax[2].plot(np.nanmean(block_score_agg[:,(isRL1HM2==2)], axis=1), label='HM')
ax[2].legend()
ax[3].plot(np.sum(~np.isnan(block_score_agg[:,(isRL1HM2==1)]), axis=1), label='RL')
ax[3].plot(np.sum(~np.isnan(block_score_agg[:,(isRL1HM2==2)]), axis=1), label='HM')
ax[3].legend()
plt.show()