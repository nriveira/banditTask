import wrangler_nick as nick
from utilities import exp_mov_ave

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

#data_loc = r'/Users/nick/Projects/ODoherty/vince_data/csv' # For mac
data_loc = 'C:/Users/nrive/Projects/banditTask/vince_code/csv' # For pc
exclude = [0,1,2,3,4,5,14,24,34,37] # Added 11, 12, 16, 17, 20, 23 for low block num

def lin_fit(x, a, b):
    return a*x+b

def exp_fit(x, a, b, c):
    return a*np.exp(b*x)+c

def get_block_score(subjectData):
    # Block reset statistics 
    aggHighChosen = subjectData['highChosen'].cumsum()

    # Adjust the cumsum to account for resetting
    rev = np.append(np.append(0, np.where(subjectData['reverseTrial']==1)),0)
    rev_i = 0
    bias = 0

    for i, v in enumerate(aggHighChosen): # For each trial
        next_rev = rev[rev_i] # Reset based on reversal
        aggHighChosen[i] = v - bias
        if(i == next_rev): # If reversal is hit, set next reversal
            bias = v
            rev_i += 1

    block_performance = np.divide(aggHighChosen, subjectData['blockTrialNo'])
    block_score = np.zeros((np.max(subjectData['blockTrialNo']), np.max(subjectData['block'])))+np.nan
    lin_fit_score = np.zeros((2, np.max(subjectData['block'])))
    exp_fit_score = np.zeros((3, np.max(subjectData['block'])))

    for b in range(np.max(subjectData['block'])):
        arr = block_performance[subjectData['block']==(b+1)].to_numpy()
        exp_avg, _ = exp_mov_ave(arr)
        block_score[:len(exp_avg),b] = exp_avg

        x = np.arange(len(exp_avg))
        if(len(exp_avg) > 3):
            lin_params, lin_covs = curve_fit(lin_fit, x, exp_avg)
            lin_fit_score[:,b] = lin_params
            #exp_params, exp_covs = curve_fit(exp_fit, x, arr, maxfev=1000000)

    return block_score, lin_fit_score

# Sample plot
sub = 6
fig, ax = plt.subplots(1,1)
block_score, lin_fit_score = get_block_score(nick.subjectDataWrangler(sub, data_loc).subjectData)
plt.plot(block_score)
plt.plot(np.nanmean(block_score, axis=1), 'k')
plt.title('Sample Aggregation of Blocks with Average')
plt.legend(['Block 1','Block 2','Block 3','Block 4','Average'])
plt.show()

# Aggregated subjects plot
fig, ax = plt.subplots(2,1, sharex=True, sharey=True)
plt.suptitle('Aggregation of Subjects')  
lin_fit_rl_agg = np.array([])
lin_fit_rl_bgg = np.array([])
lin_fit_hm_agg = np.array([])
lin_fit_hm_bgg = np.array([])

for sub in range(97):
    if(sub not in exclude):     
        # Data after 'wrangling'
        subjectData = nick.subjectDataWrangler(sub, data_loc).subjectData
        if(subjectData['block'].max() < 3):
            print(subjectData['subID'][0])

        block_score, lin_fit_score = get_block_score(subjectData)
        slope = np.array(lin_fit_score[0,:])
        intercept = np.array(lin_fit_score[1,:])

        if(subjectData['instructCond'][0]=='rl'):
            ax[0].plot(np.nanmean(block_score, axis=1), 'k', alpha=0.25)
            lin_fit_rl_agg = np.concatenate((lin_fit_rl_agg, slope))
            lin_fit_rl_bgg = np.concatenate((lin_fit_rl_bgg, intercept))
        else:
            ax[1].plot(np.nanmean(block_score, axis=1), 'b', alpha=0.25)
            lin_fit_hm_agg = np.concatenate((lin_fit_hm_agg, slope))
            lin_fit_hm_bgg = np.concatenate((lin_fit_hm_bgg, intercept))
  
ax[0].legend(['rl'])     
ax[1].legend(['hm'])
plt.show()

# Aggregated Linear Coefficients
fig, ax = plt.subplots(2,1, sharex=True, sharey=True)
plt.suptitle('Linear Fit Coefficients')
ax[0].hist(lin_fit_rl_agg, alpha = 0.5)
ax[0].hist(lin_fit_hm_agg, alpha = 0.5)
ax[0].set_title('Slope of performance across blocks')
ax[0].legend(['RL', 'HM'])
ax[1].hist(lin_fit_rl_bgg, alpha = 0.5)
ax[1].hist(lin_fit_hm_bgg, alpha = 0.5)
ax[0].set_title('Intercept of performance across blocks')
ax[1].legend(['RL', 'HM'])
plt.show()