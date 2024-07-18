from wrangler_nick import subjectDataWrangler

import numpy as np
import matplotlib.pyplot as plt

data_loc = r'/Users/nick/Projects/ODoherty/vince_data/csv' # For mac
#data_loc = 'C:/Users/nrive/Projects/banditTask/vince_data/csv' # For pc
exclude = [0,1,2,3,4,5,14,24,34,37] # Added 11, 12, 16, 17, 20, 23 for low block num
sampleSub = 6
win_size = 10

# Figure 1
plt.figure(figsize=(10,8))
def subject_trials_to_thresh(sampleSub, data_loc):
    subjectData = subjectDataWrangler(sampleSub, data_loc).subjectData
    unique, counts = np.unique(subjectData['block'], return_counts=True)
    return unique, counts

unique, counts = subject_trials_to_thresh(sampleSub, data_loc)
plt.plot(unique, counts)
plt.xlabel('Block Number')
plt.ylabel('# Trials')
plt.title('Trials per Block')
plt.show()

# Figure 2
def subject_pHighChosen(sampleSub, data_loc, win_size):
    subjectData = subjectDataWrangler(sampleSub, data_loc).subjectData
    rev = np.where(subjectData['reverseTrial']==1)[0]
    rev_centered = []
    for r in rev:
        if((r-win_size > 0) and (r+win_size < len(subjectData['highChosen']))):
            rev_centered.append((np.array(subjectData['pHighChosen'][r-win_size:r+win_size+1])))
    return rev_centered

rev_centered = subject_pHighChosen(sampleSub, data_loc, win_size)
plt.figure(figsize=(10,8))
for r in rev_centered:
    plt.plot(np.arange(-win_size,win_size+1),r)
plt.plot(np.arange(-win_size,win_size+1), np.array(rev_centered).mean(axis=0), 'b--')
plt.axvline(0, color='k')
plt.xlabel('Trials to Reversal')
plt.ylabel('aggregated p(highChosen)')
plt.title('highChosen Probability')
plt.show()

# Figure 4
def subject_pSwitch(sampleSub, data_loc, win_size):
    subjectData = subjectDataWrangler(sampleSub, data_loc).subjectData
    rev = np.where(subjectData['reverseTrial']==1)[0]
    rev_centered = []
    for r in rev:
        if((r-win_size > 0) and (r+win_size < len(subjectData['pSwitch']))):
            rev_centered.append((np.array(subjectData['pSwitch'][r-win_size:r+win_size+1])))
    return rev_centered

rev_centered = subject_pSwitch(sampleSub, data_loc, win_size)
plt.figure(figsize=(10,8))
for r in rev_centered:
    plt.plot(np.arange(-win_size,win_size+1),r)
plt.plot(np.arange(-win_size,win_size+1), np.array(rev_centered).mean(axis=0), 'b--')
plt.axvline(0, color='k')
plt.xlabel('Trials to Reversal')
plt.ylabel('p(Switch)')
plt.title('Switch Probability')
plt.show()