from wrangler_nick import subjectDataWrangler
from utilities import exp_mov_ave

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#data_loc = r'/Users/nick/Projects/ODoherty/vince_data/csv' # For mac
data_loc = 'C:/Users/nrive/Projects/banditTask/vince_data/csv' # For pc
exclude = [0,1,2,3,4,5,14,24,34,37] # Added 11, 12, 16, 17, 20, 23 for low block num
sampleSub = 8

class block_stats():
    def __init__(self, subjectData):
        self.subjectData = subjectData
        self.get_probs()
        self.get_block_agg()

    def get_probs(self):
        # Block reset statistics 
        subjectData = self.subjectData
        highChosenCumSum = np.zeros((len(subjectData['highChosen'])))
        isWinCumSum = np.zeros((len(subjectData['isWin']),))
        highChosenStim1CumSum = np.zeros((len(subjectData['stim1_isWin']),))
        highChosenStim2CumSum = np.zeros((len(subjectData['stim2_isWin']),))

        # Adjust the cumsum to account for resetting
        rev = np.where(subjectData['reverseTrial']==1)[0]

        last_rev = 0
        for r in rev:
            exp_hccs, _ = exp_mov_ave(subjectData['highChosen'][last_rev:r+1].cumsum())
            exp_iwcs, _ = exp_mov_ave(subjectData['isWin'].fillna(0)[last_rev:r+1].cumsum())
            exp_s1cs, _ = exp_mov_ave(subjectData['stim1_isWin'].fillna(0)[last_rev:r+1].cumsum())
            exp_s2cs, _ = exp_mov_ave(subjectData['stim2_isWin'].fillna(0)[last_rev:r+1].cumsum())

            highChosenCumSum[last_rev:r+1] = exp_hccs
            isWinCumSum[last_rev:r+1] = exp_iwcs
            highChosenStim1CumSum[last_rev:r+1] = exp_s1cs
            highChosenStim2CumSum[last_rev:r+1] = exp_s2cs
            last_rev = r+1

        self.highChosen = highChosenCumSum #np.divide(, subjectData['blockTrialNo'])
        self.isWin = isWinCumSum #np.divide(, subjectData['blockTrialNo'])

    def get_block_agg(self):
        subjectData = self.subjectData
        block_score = np.zeros((np.max(subjectData['blockTrialNo']), np.max(subjectData['block'])))+np.nan
        for b in range(np.max(subjectData['block'])):
                arr = self.highChosen[subjectData['block']==(b+1)]
                block_score[:len(arr),b] = arr
        self.block_score = block_score


# Show sample result
subjectData = subjectDataWrangler(sampleSub, data_loc).subjectData
sBlock = block_stats(subjectData)

plt.figure(figsize=(20,8))
plt.plot(sBlock.highChosen, label='p(High Chosen)')
plt.plot(sBlock.isWin, label='p(isWin)')
plt.legend()
plt.title('Single Subject Across all Trials for subject {}'.format(sBlock))
plt.show()

plt.figure(figsize=(20,8))
plt.plot(sBlock.block_score, 'r', alpha=0.5)
plt.plot(np.nanmean(sBlock.block_score, axis=1),'k--')
plt.title('Single Subject {} Collapsed Across Block'.format(sampleSub))
plt.show()