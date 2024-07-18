from descriptive_nick import block_stats
from conditional_nick import block_condStats
from wrangler_nick import subjectDataWrangler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_loc = r'/Users/nick/Projects/ODoherty/vince_data/csv' # For mac
#data_loc = 'C:/Users/nrive/Projects/banditTask/vince_data/csv' # For pc
exclude = [0,1,2,3,4,5,14,24,34,37] # Added 11, 12, 16, 17, 20, 23 for low block num

samSub = 8
samSubjectData = subjectDataWrangler(samSub, data_loc).subjectData
block_score_agg = np.zeros((len(samSubjectData['block']), 97))+np.nan
isRL1HM2 = np.zeros((97))

fig, ax = plt.subplots(2,1,figsize=(20,8))
for sub in range(97):
    if(sub not in exclude):     
        # Data to 'wrangle'
        subjectData = subjectDataWrangler(sub, data_loc).subjectData
        block = block_stats(subjectData)
        sBlock = block_condStats(subjectData)
        if(subjectData['instructCond'][0]=='rl'):
            isRL1HM2[sub] = 1
        elif(subjectData['instructCond'][0]=='hm'):
            isRL1HM2[sub] = 2

        # Get trial types
        trials_type1 = np.where(sBlock.stimHigh==1)[0]
        trials_type2 = np.where(sBlock.stimHigh==0)[0]

        block_score_agg[:len(sBlock.condProb),sub] = np.nanmean(sBlock.condProb, axis=1)

        ax[0].plot(sBlock.condProb[:,trials_type1])
        ax[0].set_title('Stim 1')
        ax[1].plot(sBlock.condProb[:,trials_type2])
        ax[1].set_title('Stim 2')

# plt.subplots(2,1,figsize=(20,8))
# ax[0].plot(np.mean(block_score_agg[:,np.where(isRL1HM2==1)]))
# ax[1].plot(block_score_agg[:,np.where(isRL1HM2==2)])