from wrangler_nick import subjectDataWrangler

import numpy as np
import matplotlib.pyplot as plt

data_loc = r'/Users/nick/Projects/ODoherty/vince_data/csv' # For mac
#data_loc = 'C:/Users/nrive/Projects/banditTask/vince_data/csv' # For pc
exclude = [0,1,2,3,4,5,14,24,34,37] # Added 11, 12, 16, 17, 20, 23 for low block num
sampleSub = 6
win_size = 10

class kirs_subject_nick():
    def __init__(self, subjectData):
        self.subjectData = subjectData

    # Figure 1
    def subject_trials_to_thresh(self):
        unique, counts = np.unique(self.subjectData['block'], return_counts=True)
        return unique, counts

    # Figure 2
    def subject_pHighChosen(self, win_size):
        rev = np.where(self.subjectData['reverseTrial']==1)[0]
        rev_centered = []
        for r in rev:
            if((r-win_size > 0) and (r+win_size < len(self.subjectData['highChosen']))):
                rev_centered.append((np.array(self.subjectData['pHighChosen'][r-win_size:r+win_size+1])))
        return rev_centered

    def subject_pHighGivenB(self, win_size):
        rev = np.where(self.subjectData['reverseTrial']==1)[0]
        rev_stim = []
        for r in rev:
            if((r-win_size > 0) and (r+win_size < len(self.subjectData['highChosen']))):
                if(subjectData['stimHigh'][r+1] == 1):
                    rev_stim.append((1-np.array(self.subjectData['pHighChosen'][r-win_size:r+win_size+1])))
                elif(subjectData['stimHigh'][r+1] == 0):
                    rev_stim.append((np.array(self.subjectData['pHighChosen'][r-win_size:r+win_size+1])))
        return rev_stim

    # Figure 4
    def subject_pSwitch(self, win_size):
        rev = np.where(subjectData['reverseTrial']==1)[0]
        rev_centered = []
        for r in rev:
            if((r-win_size > 0) and (r+win_size < len(subjectData['pSwitch']))):
                rev_centered.append((np.array(subjectData['switch'][r-win_size:r+win_size+1])))
        return rev_centered


# Plotting
# Single Subject Plots
subjectData = subjectDataWrangler(sampleSub, data_loc).subjectData
kirs = kirs_subject_nick(subjectData)

# Figure 1S
plt.figure(figsize=(10,8))
unique, counts = kirs.subject_trials_to_thresh()
plt.plot(unique, counts)
plt.xlabel('Block Number')
plt.ylabel('# Trials')
plt.title('Trials per Block')
plt.show()

# Figure 2S
plt.figure(figsize=(10,8))
rev_centered = kirs.subject_pHighChosen(win_size)
for r in rev_centered:
    plt.plot(np.arange(-win_size,win_size+1),r)
plt.plot(np.arange(-win_size,win_size+1), np.array(rev_centered).mean(axis=0), 'b--')
plt.axvline(0, color='k')
plt.xlabel('Trials to Reversal')
plt.ylabel('aggregated p(highChosen)')
plt.title('highChosen Probability')
plt.show()

# Figure 3S
plt.figure(figsize=(10,8))
rev_cond = kirs.subject_pHighGivenB(win_size)
for r in rev_cond:
    plt.plot(np.arange(-win_size,win_size+1),r)
plt.plot(np.arange(-win_size,win_size+1), np.array(rev_cond).mean(axis=0), 'b--')
plt.axvline(0, color='k')
plt.xlabel('Trials to Reversal')
plt.ylabel('p(highChosen|B High)')
plt.title('probability given B')
plt.show()

# Figure 4S
plt.figure(figsize=(10,8))
rev_centered = kirs.subject_pSwitch(win_size)
for r in rev_centered:
    plt.plot(np.arange(-win_size,win_size+1),r)
plt.plot(np.arange(-win_size,win_size+1), np.array(rev_centered).mean(axis=0), 'b--')
plt.axvline(0, color='k')
plt.xlabel('Trials to Reversal')
plt.ylabel('p(Switch)')
plt.title('Switch Probability')
plt.show() 