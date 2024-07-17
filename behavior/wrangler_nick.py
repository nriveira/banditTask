import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Index individual plots based on csv name format

columnDatatype = {# Session Metadata
                  'sessNo' : 'Int8',
                  'subID' : 'Int8',  
                  'block' : 'Int8',
                  'blockTrialNo' : 'Int8',
                  'blockBin' : 'Int8',
                  'instructCond' : 'string',

                  # Single Trial Variables
                  'stimHigh' : 'Int8',
                  'reverseTrial' : 'Int8',
                  'responseKey' : 'Int8',
                  'RT' : 'Float32',

                  # Trial Result Derivatives
                  'highChosen' : 'Int8',
                  'isWin' : 'Int8',
                  'stim1_isWin' : 'Int8',
                  'stim2_isWin' : 'Int8',
                  'payOut' : 'Float32',
                  'accum_payOut' : 'Float32'}

class subjectDataWrangler():
    def __init__(self, sub, data_loc):
        # Fix strings
        data_string = '/sub{}_data.csv'.format(sub)

        # Will add onset string info later, not needed now
        onset_string = '/sub{}_onsets.csv'.format(sub)
        sub_data_loc = data_loc+data_string

        # Load in CSV
        sub_rawData = pd.read_csv(sub_data_loc)
        # Select certain columns
        sub_data = sub_rawData[list(columnDatatype.keys())].copy(deep=True)

        # Data cleansing
        # Map binary variables before assigning them (Be careful of mappings!)
        sub_data['stimHigh'] = sub_data['stimHigh'].map({2:1, 
                                                         1:0})
                                                         
        sub_data['switch'] = np.abs(np.append(0, np.diff(sub_data['responseKey'].map({4:1,1:0}))))
        self.subjectData = sub_data.astype(columnDatatype)

# Sample uses of this function
sub = 6
#data_loc = r'/Users/nick/Projects/ODoherty/vince_data/csv' # For mac
data_loc = 'C:/Users/nrive/Projects/banditTask/vince_data/csv' # For pc

subjectData = subjectDataWrangler(sub, data_loc).subjectData
subjectData.plot(figsize=(24, 24), subplots=True, sharex=True)