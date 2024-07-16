# Imports and variable setting
import wrangler_nick as nick
import pandas as pd
import matplotlib.pyplot as plt

sub = 8
#data_loc = r'/Users/nick/Projects/ODoherty/vince_data/csv' # For mac
data_loc = 'C:/Users/nrive/Projects/banditTask/vince_code/csv' # For pc

# Data overview after 'wrangling' 
subjectData = nick.subjectDataWrangler(sub, data_loc).subjectData
subjectData.plot(figsize=(24, 24), subplots=True, sharex=True)

# Exploring conditional probabilities of task performance
# Split by wins

columns=['reverseTrial', 'responseKey', 'highChosen', 'isWin']
for col in columns:
    for title, group in subjectData.groupby(col):
        group.hist(figsize=(20, 8), 
                column=['reverseTrial', 'responseKey', 'RT', 'highChosen', 'isWin', 'payOut', 'accum_payOut'])
        if(title==1):
            plt.suptitle(col)
        else:
            plt.suptitle(('not '+col))