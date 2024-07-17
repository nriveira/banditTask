#%% Imports
import numpy as np 
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import json
import matplotlib.pyplot as plt

import utilities as utils

#%% Parse traces 
def parse_traces(subList, datDir, jsonDir):
    # Create empty dataframe
    group_dict = {}
    
    # Loop through subjects
    for subID in subList:
         # Load subject dataframe
        subDf = pd.read_csv(f'{datDir}/sub{subID}_data.csv')
        
        sub_dict = {}
        
        # Split subDf by session  
        for sIdx, sessID in enumerate(subDf.sessNo.unique()):
            sess_path = f'{jsonDir}/sub{subID}_sess{sessID}_data.json'
            
            # Load in json file
            with open(sess_path, "r") as f:
                sessJson = json.load(f)
                sessDat = sessJson['sessionInfo'][sIdx]
                
            hand_position_trials = sessDat['hand_position_trials']
            hand_velocity_trials = sessDat['hand_velocity_trials']
            object_position_trials = sessDat['object_position_trials']
            object_velocity_trials = sessDat['object_velocity_trials']
            
            sess_trajectory_dict = {'hand_position_trials': hand_position_trials,
                                    'hand_velocity_trials': hand_velocity_trials,
                                    'object_position_trials': object_position_trials,
                                    'object_velocity_trials': object_velocity_trials}
            
            sess_trajectory_preproc, fig_preproc = preprocess_traces(subID, sessID, sess_trajectory_dict)
        
            sub_dict[int(sessID)] = sess_trajectory_preproc
    
        group_dict[int(subID)] = sub_dict
            
    return group_dict


#%% Preprocess traces
def preprocess_traces(sub, sess, data, vel_thr=0.1):
    # Identify number of trials
    numTrials = len(data['hand_position_trials'])
    
    # Iterate over trials for preproc     
    hand_position_trials = data['hand_position_trials']
    hand_velocity_trials = data['hand_velocity_trials']
    hand_position_preproc = [] 
    hand_velocity_preproc = []
    isNoise = np.zeros(numTrials, dtype=bool)
    
    for t in range(numTrials):
        pos = hand_position_trials[t]
        vel = hand_velocity_trials[t]
                
        if pos:             
            # Identify frames with sudden shifts in velocity            
            frame_shifts = np.abs(np.roll(np.diff(np.sign(vel),1),1))            
            shift_idx = np.where(frame_shifts>1)[0]
            
            if shift_idx.shape[0] > 0:
                # Identify period around shift (2 frames)
                start_idx = shift_idx.min() - 2
                end_idx = shift_idx.max() + 2

                # Evaluate whether to interpolate shift 
                interpolate = False 
                interpolate_idx = []
                for i in range(start_idx, end_idx):
                    if np.abs(vel[i]) > vel_thr: # if velocity greater than thr
                        interpolate_idx.append(i)
                        interpolate = True

                # Linear interpolate   
                if interpolate:    
                    isNoise[t] = 1
                    # Identify start and end of interpolation
                    interp_start_idx = interpolate_idx[0] - 2
                    interp_end_idx = interpolate_idx[-1] + 2
                    
                    # Linear interpolation of extended period around shift
                    pos_interp = utils.linear_interp(interp_start_idx, interp_end_idx, pos)
                                                                                                                                                                                                                                      
                    # Compute velocity of preproc position vector
                    vel_interp = np.diff(pos_interp)
                    # Append to preproc vectors
                    hand_position_preproc.append(pos_interp)
                    hand_velocity_preproc.append(vel_interp)
                else: 
                    hand_position_preproc.append(pos)
                    hand_velocity_preproc.append(vel)
            else:
                hand_position_preproc.append(pos)
                hand_velocity_preproc.append(vel)
        else:
            hand_position_preproc.append([])
            hand_velocity_preproc.append([])
            
    # Create preproc dictionary
    data_preproc = {'hand_position_trials': hand_position_preproc,
                    'hand_velocity_trials': hand_velocity_preproc,
                    'object_position_trials': data['object_position_trials'],
                    'object_velocity_trials': data['object_velocity_trials'],
                    'is_noisy_trials': isNoise.tolist()}
   
    # Make figure of preprocessing 
        
    fig, ax = plt.subplots(2,2, figsize=(3,3), sharey=False)
    ax = ax.flatten()
    fig.suptitle(f'Subject {sub}, Session {sess}')

    for t in range(40):    
        ax[0].plot(data['hand_position_trials'][t], c='blue', alpha=0.5)
        ax[1].plot(data['hand_velocity_trials'][t], c='blue', alpha=0.5)
        ax[2].plot(data_preproc['hand_position_trials'][t], c='green', alpha=0.5)
        ax[3].plot(data_preproc['hand_velocity_trials'][t], c='green', alpha=0.5)
        ax[2].set_ylim(ax[0].get_ylim())
        ax[3].set_ylim(ax[1].get_ylim())
    plt.tight_layout()
    return data_preproc, fig


#%% Function to add "is noisy trial" covariate 
# estimated from trajectory preprocessing  into dataframe 
def add_noise_covar(groupDf, group_traj):

    # Iterate over subjects 
    for subID, sessID in groupDf[['subID','sessNo']].drop_duplicates().values:
        # Get session dataframe 
        sessDf = groupDf[(groupDf.subID == subID) & (groupDf.sessNo == sessID)]
        
        # Get vector of whether trial is noisy 
        sess_is_noisy = group_traj[subID][sessID]['is_noisy_trials'] 
        
        # Put into group dataframe
        groupDf.loc[(groupDf.subID == subID) & (groupDf.sessNo == sessID), 'is_noisy_trial'] = sess_is_noisy
    return groupDf


#%% Preprocess rt start times
def preprocess_rt_start(groupDf, group_traj, pos_thr=0.1):

    # Iterate over subjects 
    for subID, sessID in groupDf[['subID','sessNo']].drop_duplicates().values:
        # Get session dataframe 
        sessDf = groupDf[(groupDf.subID == subID) & (groupDf.sessNo == sessID)]
        # Get original rt start times
        rt_start = sessDf.RT_start.values
        # Get preprocessed hand position and velocity
        sess_hand_pos = group_traj[subID][sessID]['hand_position_trials'] 
        
        # Iterate over trials in session
        new_rt_start = np.ones(len(rt_start), dtype=float) * np.nan
        for t in range(len(rt_start)):
            if not np.isnan(rt_start[t]):
                # Get hand position for trial
                pos = sess_hand_pos[t]
                # Compute difference relative to start position
                pos_diff = np.abs(np.array(pos) - np.mean(pos[0:5]))
                # Check if any position differences are greater than threshold
                if np.any(pos_diff > pos_thr):
                    # Identify first frame where hand position is greater than threshold
                    start_idx = np.where(pos_diff > pos_thr)[0][0]
                    # Convert from frame to time 
                    new_rt_start[t] = start_idx / 60
                    
            
        # Put into group dataframe
        groupDf.loc[(groupDf.subID == subID) & (groupDf.sessNo == sessID), 'RT_start_preproc'] = new_rt_start                                            
    return groupDf

#%% Create group dataframe
def create_group_df(subList, datDir, n_back=10):
    """
    Create a group dataframe from the individual subject dataframes
    """
    # Create empty dataframe
    groupDf = []
    
    # Loop through subjects
    for subID in subList:
        # Load subject dataframe
        subDf = pd.read_csv(f'{datDir}/sub{subID}_data.csv')
        
        # Recode isHit to isMiss
        subDf['isMiss'] = 1 - subDf.isHit
        
        # Recode response
        subDf['response_right'] = subDf.responseKey
        subDf['response_left'] = 1 - subDf.responseKey
        
        # Split subDf by session and 
        for sessID in subDf.sessNo.unique():
            sessDf = subDf[subDf.sessNo == sessID]
            
            # Preprocess rt start times using traces (robust against small joystick perturbations)
            
            # apply exponential moving average to isHit
            convHit, _ = utils.exp_mov_ave(sessDf.isHit)
            sessDf.loc[:, 'convHit'] = convHit
            
            # apply exponential moving average to highChosen
            convHigh, _ = utils.exp_mov_ave(sessDf.highChosen)
            sessDf.loc[:, 'convHigh'] = convHigh
            
            # apply exponential moving average to outcome
            convOut, _ = utils.exp_mov_ave(sessDf.payOut)
            sessDf.loc[:, 'convOut'] = convOut
            
            # encode switches
            sessDf.loc[:, 'switch'] = sessDf.response_stimID.diff()
            
            
            # Get n-back hit and high choice
            for n in range(1,n_back+1):
                # Previous hit 
                n_back_hit = sessDf.isHit.shift(n)
                n_back_miss = sessDf.isMiss.shift(n)
                sessDf.loc[:, f'isHit_{n}_back'] = n_back_hit
                sessDf.loc[:, f'isMiss_{n}_back'] = n_back_miss
                
                # Previous high
                n_back_high = sessDf.highChosen.shift(n)
                sessDf.loc[:, f'highChosen_{n}_back'] = n_back_high
                
                # Previous outcome
                n_back_outcome = sessDf.payOut.shift(n)
                sessDf.loc[:, f'win_{n}_back'] = n_back_outcome
                sessDf.loc[:, f'loss_{n}_back'] = 1 - n_back_outcome
                
                # Split losses by misses vs hits
                sessDf.loc[:, f'loss_miss_{n}_back'] = sessDf[f'loss_{n}_back'] * sessDf[f'isMiss_{n}_back']
                sessDf.loc[:, f'loss_hit_{n}_back'] = sessDf[f'loss_{n}_back'] * sessDf[f'isHit_{n}_back']
                
                # Previous selected stims
                n_back_stim1 = sessDf.selected_stim1.shift(n)
                sessDf.loc[:, f'selected_stim1_{n}_back'] = n_back_stim1                
                n_back_stim2 = sessDf.selected_stim2.shift(n)
                sessDf.loc[:, f'selected_stim2_{n}_back'] = n_back_stim2
                
                
                # Previous response key
                n_back_respR = sessDf.response_right.shift(n)
                sessDf.loc[:, f'response_right_{n}_back'] = n_back_respR
                n_back_respL = sessDf.response_left.shift(n)
                sessDf.loc[:, f'response_left_{n}_back'] = n_back_respL
                

            # Drop index
            sessDf.reset_index(drop=True, inplace=True)
            
            # Append to group dataframe
            groupDf.append(sessDf)
    
    # Concatenate group dataframe
    return pd.concat(groupDf, ignore_index=True)

#%% Posterior predictive group dataframe
def create_group_df_postpred(subList, datDir, modOutDir, model, n_back=10):
    """
    Create a group dataframe from the individual subject dataframes
    """
    # Create empty dataframe
    groupDf = []
    
    # Loop through subjects
    for subID in subList:
        # Load subject dataframe
        subDf = pd.read_csv(f'{datDir}/sub{subID}_data.csv')
        
        # Load in model predictions
        modDf = pd.read_csv(f'{modOutDir}/sub{subID}_{model}_postPred.csv')        
        
        # Identify model correct choice and stims chosen
        modDf['highChosen'] = (modDf['simChoice_stimID'] == subDf['stimHigh'])
        modDf['selected_stim1'] = modDf['simChoice_stimID'] == 1
        modDf['selected_stim2'] = modDf['simChoice_stimID'] == 2
        modDf['stim1_High'] = subDf['stim1_High']
        
        # Get a few columns from the subject data
        modDf[['subID','sessNo','trialNo','payOut']] = subDf[['subID','sessNo','trialNo','payOut']]
        
        # Get response key based on order
        modDf['order'] = (subDf.leftStim == 2).astype(int) # If order == 0, stim1 was on the left 
        # Recode response
        modDf['response_right'] = (modDf.order == 0) & (modDf.simChoice == 0)
        modDf['response_left'] = (modDf.order == 0) & (modDf.simChoice == 1)
        
        modDf['isHit'] = subDf['isHit']
        
                        
        # Split subDf by session and 
        for sessID in modDf.sessNo.unique():
            sessDf = modDf[modDf.sessNo == sessID]
                        
            # apply exponential moving average to highChosen
            convHigh, _ = utils.exp_mov_ave(sessDf.highChosen)
            sessDf.loc[:, 'convHigh'] = convHigh
            
            # apply exponential moving average to outcome
            convOut, _ = utils.exp_mov_ave(sessDf.payOut)
            sessDf.loc[:, 'convOut'] = convOut
            
            # encode switches
            sessDf.loc[:, 'switch'] = sessDf.simChoice_stimID.diff()
            
            # # Get binary hit expectations
            # simHit_median = sessDf['simHit'].median()                                
            # sessDf['isHit'] = np.where(sessDf['simHit'] < simHit_median, 1, 0)
        
            # Get n-back hit and high choice
            for n in range(1,n_back+1):
                # Previous hit 
                n_back_hit = sessDf.isHit.shift(n)
                sessDf.loc[:, f'isHit_{n}_back'] = n_back_hit
                
                # Previous high
                n_back_high = sessDf.highChosen.shift(n)
                sessDf.loc[:, f'highChosen_{n}_back'] = n_back_high
                
                # Previous outcome
                n_back_outcome = sessDf.payOut.shift(n)
                sessDf.loc[:, f'outcome_{n}_back'] = n_back_outcome
                
                # Previous selected stims
                n_back_stim1 = sessDf.selected_stim1.shift(n)
                sessDf.loc[:, f'selected_stim1_{n}_back'] = n_back_stim1                
                n_back_stim2 = sessDf.selected_stim2.shift(n)
                sessDf.loc[:, f'selected_stim2_{n}_back'] = n_back_stim2
                
                # Previous response key
                n_back_respR = sessDf.response_right.shift(n)
                sessDf.loc[:, f'response_right_{n}_back'] = n_back_respR
                n_back_respL = sessDf.response_left.shift(n)
                sessDf.loc[:, f'response_left_{n}_back'] = n_back_respL
                
             
            # Drop index
            sessDf.reset_index(drop=True, inplace=True)
            
            # Append to group dataframe
            groupDf.append(sessDf)
    
    # Concatenate group dataframe
    return pd.concat(groupDf, ignore_index=True)



