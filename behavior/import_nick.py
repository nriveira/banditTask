#%% 
# Import Statements/Load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Index individual plots based on csv name format
sub = 8
data_loc = r'/Users/nick/Projects/ODoherty/vince_data/csv' # For mac
#data_loc = 'C:/Users/nrive/Projects/banditTask/vince_data/csv' # For pc

data_string = '/sub{}_data.csv'.format(sub)
onset_string = '/sub{}_onsets.csv'.format(sub)

#%% Find CSVs to import
sample_data_loc = data_loc+data_string
sample_onset_loc = data_loc+onset_string

# Import the individual data
sample_data = pd.read_csv(sample_data_loc, 
                          true_values=['early', 'True'], 
                          false_values=['late', 'False'])
                    
sample_offset = pd.read_csv(sample_onset_loc, 
                            true_values=['early', 'True'], 
                            false_values=['late', 'False'])

sample_data.plot(figsize=(24, 24), subplots=True, sharex=True, layout=(11,2), xlabel='Trial Number')
plt.show()

for title, group in sample_data.groupby(['stim1_pWin']):
    group.hist(figsize=(20, 8))
    plt.suptitle(('Stim1 Probability: ' + str(title[0])))
    plt.tight_layout()