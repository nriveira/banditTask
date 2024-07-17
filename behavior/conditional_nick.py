from wrangler_nick import subjectDataWrangler
from utilities import exp_mov_ave

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_loc = r'/Users/nick/Projects/ODoherty/vince_data/csv' # For mac
#data_loc = 'C:/Users/nrive/Projects/banditTask/vince_data/csv' # For pc
exclude = [0,1,2,3,4,5,14,24,34,37] # Added 11, 12, 16, 17, 20, 23 for low block num
sampleSub = 8

class block_condStats():
    def __init__(self, subjectData):
        self.subjectData = subjectData
        self.get_cond_probs() # Get trial fits for each subject
        # self.get_block_agg() # Get aggregate data of all blocks

    def get_cond_probs(self):
        # Block reset statistics 
        subjectData = self.subjectData
        rev = np.where(subjectData['reverseTrial']==1)[0]
        highChosenCond = np.zeros((len(subjectData['highChosen']), len(rev)))
        stimHigh = np.zeros((len(rev)))

        # Adjust the cumsum to account for resetting
        last_rev = 0
        for i,r in enumerate(rev):
            condTrials = subjectData['highChosen'][last_rev:r+1].dropna()
            exp_ct, _ = exp_mov_ave(condTrials.cumsum())

            highChosenCond[:len(exp_ct),i] = exp_ct
            highChosenCond[len(exp_ct):,i] = np.nan
            stimHigh[i] = subjectData['stimHigh'][last_rev]
            
            last_rev = r+1

        self.condProb = highChosenCond
        self.stimHigh = stimHigh

# Show sample result
subjectData = subjectDataWrangler(sampleSub, data_loc).subjectData
sBlock = block_condStats(subjectData)

fig, ax = plt.subplots(2,1,figsize=(20,8), sharex=True, sharey=True)
trials_type1 = np.where(sBlock.stimHigh==1)[0]
trials_type2 = np.where(sBlock.stimHigh==0)[0]

plt.figure(figsize=(20,8))
ax[0].plot(sBlock.condProb[:,trials_type1])
ax[0].set_title('Subject {} Stim 1 High'.format(sampleSub))
ax[1].plot(sBlock.condProb[:,trials_type2])
ax[1].set_title('Subject {} Stim 2 High'.format(sampleSub))
plt.show()