from kirs_nick import kirs_subject_nick
from wrangler_nick import subjectDataWrangler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_loc = r'/Users/nick/Projects/ODoherty/vince_data/csv' # For mac
#data_loc = 'C:/Users/nrive/Projects/banditTask/vince_data/csv' # For pc
exclude = [0,1,2,3,4,5,14,24,34,37] # Added 11, 12, 16, 17, 20, 23 for low block num

agg_RLcounts = []
agg_HMcounts = []
agg_rev_centered = []
agg_rev_stim = []
agg_pSwitch = []

for sub in range(97):
    if(sub not in exclude):     
        # Data to 'wrangle'
        subjectData = subjectDataWrangler(sub, data_loc).subjectData
        kirs = kirs_subject_nick(subjectData)
        unique, counts = kirs.subject_trials_to_thresh()
        if(subjectData['instructCond'][0]=='rl'):
            agg_RLcounts.append((counts))
        elif(subjectData['instructCond'][0]=='hm'):
            agg_HMcounts.append((counts))

# Figure 1 Aggregate
rlAggs = np.zeros((10, len(agg_RLcounts))) + np.nan
hmAggs = np.zeros((10, len(agg_HMcounts))) + np.nan

for i in range(len(agg_RLcounts)):
    rlAggs[:len(agg_RLcounts[i]),i] = agg_RLcounts[i]

for i in range(len(agg_HMcounts)):
    hmAggs[:len(agg_HMcounts[i]),i] = agg_HMcounts[i]

fig, ax = plt.subplots(3,1,figsize=(20,8), sharex=True)
ax[0].plot(rlAggs, 'r', alpha=0.5)
ax[0].plot(hmAggs, 'b', alpha=0.5)
ax[0].set_ylabel('# of Subjects')
ax[1].plot(np.nanmean(rlAggs, axis=1), 'r')
ax[1].plot(np.nanmean(hmAggs, axis=1), 'b')
ax[1].set_ylabel('Average Across Subjects')
ax[2].plot(np.nanvar(rlAggs, axis=1), 'r')
ax[2].plot(np.nanvar(hmAggs, axis=1), 'b')
ax[2].set_xlabel('Block #')
ax[2].set_ylabel('Per Block Variance')