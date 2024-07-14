#%% Import packages
import numpy as np 
import json
import pickle
from copy import copy
                    
#%% Utility functions
def exp_mov_ave(data, tau = 8, initValue = 0.5, alpha = None):
    'Exponential moving average for 1d data.  The decay of the exponential can either be specified with a time constant tau or a learning rate alpha.'
    if not alpha: alpha = 1. - np.exp(-1./tau)
    mov_ave = np.zeros(np.size(data)+1)
    mov_ave[0] = initValue
    for i, x in enumerate(data):
        if not np.isnan(x):
            mov_ave[i+1] = (1.-alpha)*mov_ave[i] + alpha*x 
        else: 
            mov_ave[i+1] = mov_ave[i]
    return mov_ave[1:], mov_ave[:-1]


#%% Utility saving functions
def convert(x):
    if hasattr(x, "tolist"):  # numpy arrays have this
      return x.tolist()
    raise TypeError(x)

def save_json(obj, name):
    with open(f'{name}.json',"w") as f:
        json.dump(obj,f, default=convert)

def load_json(name):
    with open(f'{name}.json',"r") as f:
        obj = json.load(f)
    return obj

def save_pkl(obj, name):
    # Open a file for writing in binary mode
    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump(obj, f)
        
def load_pkl(name):
    # Open a file for reading in binary mode
    with open(f'{name}.pkl', 'rb') as f:
        # Load the pickled object using the pickle.load() function
        obj = pickle.load(f)
        return obj
    
#%% Interpolation functions
def linear_interp(start_idx, end_idx, pos):
    pos_interp = copy(pos)            
    pos_interp[start_idx:end_idx] = np.linspace(pos[start_idx], 
                                                    pos[end_idx], 
                                                    end_idx-start_idx) 
    return pos_interp
