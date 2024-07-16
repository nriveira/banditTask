#%%
# Imports and variable setting
import wrangler_nick as nick

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sub = 8
#data_loc = r'/Users/nick/Projects/ODoherty/vince_data/csv' # For mac
data_loc = 'C:/Users/nrive/Projects/banditTask/vince_code/csv' # For pc

# Data overview after 'wrangling' 
subjectData = nick.subjectDataWrangler(sub, data_loc).subjectData

# %%
# Iterate through series
iter_column = 'responseKey'
window_lengths =[10]
window_iters_win = []
window_iters_loss = []

for i in window_lengths:
    columnNameWin = f'nBack{i}AWin'
    columnNameLoss = f'nBack{i}BWin'

    window_iters_win.append(columnNameWin)
    window_iters_loss.append(columnNameLoss)

    nWinsBack = pd.Series()
    nLossBack = pd.Series()

    for index, value in subjectData[iter_column].items():
        if(index >= i):
            n_backWin = subjectData['isWin'][index-i:index] == 1
            n_backLoss = subjectData['isWin'][index-i:index] == 0

            winTrials = np.sum(n_backWin)
            lossTrials = np.sum(n_backLoss)

            nWinsBack[index] = winTrials
            nLossBack[index] = lossTrials

    subjectData[columnNameWin] = nWinsBack/i
    subjectData[columnNameLoss] = nLossBack/i

# Create a new series for last n wins
for i in window_lengths:
    columnNameA = f'nBack{i}AWin'
    subjectData[columnNameA].plot()
plt.plot(subjectData['stimHigh'].map({1:1, 2:0}), '--k')
plt.title('Conditional Probabilities (A WIN)')
plt.legend(loc=5,bbox_to_anchor=(1.35, 0.5))
plt.show()

# Create a new series for last n losses
for i in window_lengths:
    columnNameB = f'nBack{i}BWin'
    subjectData[columnNameB].plot()
plt.plot(subjectData['stimHigh'].map({1:1, 2:0}), '--k')
plt.title('Centered on B Win')
plt.legend(loc=5,bbox_to_anchor=(1.35, 0.5))
plt.show()

subjectData['nBackProbWin'] = np.nansum(np.array(subjectData[window_iters_win]), axis=1)
subjectData['nBackProbLoss'] = np.nansum(np.array(subjectData[window_iters_loss]), axis=1)
plt.plot(subjectData['nBackProbWin'])
plt.plot(subjectData['nBackProbLoss'])
plt.title('Centered on A Win')
plt.legend(['pAWin', 'pBWin'], loc=5,bbox_to_anchor=(1.25, 0.5))
plt.plot(subjectData['stimHigh'].map({1:1, 2:0}), '--k')

