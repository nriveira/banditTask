from descriptive_nick import block_stats
from conditional_nick import block_condStats
from wrangler_nick import subjectDataWrangler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_loc = r'/Users/nick/Projects/ODoherty/vince_data/csv' # For mac
#data_loc = 'C:/Users/nrive/Projects/banditTask/vince_data/csv' # For pc
exclude = [0,1,2,3,4,5,14,24,34,37] # Added 11, 12, 16, 17, 20, 23 for low block num

for sub in range(97):
    if(sub not in exclude):     
        # Data after 'wrangling'
        subjectData = nick.subjectDataWrangler(sub, data_loc).subjectData
        block = block_stats(subjectData)
        blockCond = block_condStats(subjectData)