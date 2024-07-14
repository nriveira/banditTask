#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 21:13:33 2022

@author: vman
"""
#%% Import modules
import os
import numpy as np

#%% Set up directories
# Ensure that relative paths start from the same directory as this script
homeDir = '/Users/nick/Projects/ODoherty/behavior'
analysisDir = f'{homeDir}/analysis'
datDir = f'{homeDir}/data/raw/fmri/csv'

# Import custom modules
os.chdir(f'{analysisDir}/behaviour/fmri_group')
from functions import plot_reward_beh, stats_reward_beh
import functions.utilities as utils
import functions.preproc as preproc

#%% Specify experiment info
subList = np.loadtxt(f'{analysisDir}/parsing/sublist_fmri.txt').astype(int)
numSubs = len(subList)

# Parse group data 
groupDf = preproc.create_group_df(subList, datDir)

#%% Plot rewards
reward_plot = plot_reward_beh.PlotRewardBeh(groupDf)

#%% Within-session reward learning curves
fig_sub_high, fig_group_high, fig_early_late_high  = reward_plot.within_session_high() 

#%% Across-session reward learning
fig_task_sub_high, fig_task_group_high = reward_plot.across_session_high()    

#%% Plot effect of hit history on correct choice
fig_hit_high = reward_plot.hitHist_outHist_high()
        
#%% Statistics for rewards
reward_stats = stats_reward_beh.StatsRewardBeh(groupDf)
            
#%% Effect of trial number on correct choice
stats_high, fig_stats_high = reward_stats.trial_predict_high()

#%% Interaction between hit history and outcome history on current correct choice
stats_hit_out_high, fig_stats_hit_out_high = reward_stats.hit_out_interact_high()

# %% Effect of previous trial hits on correct choice
stats_past_hit_high, fig_stats_past_hit_high = reward_stats.past_hit_high()

# %% Effect of previous trial outcomes on correct choice
stats_past_out_high, fig_stats_past_out_high = reward_stats.past_out_high()

#%% Effect of previous trial outcomes conditioned on previous choice on current chosen stim
fig_conditioned_outcome_stim = reward_stats.conditioned_outcome_stim()

#%% Now condition both on previous hits and whether the previous choice was correct, on current chosen stim
fig_conditioned_hit_corr_stim = reward_stats.conditioned_hit_correct_stim()

# %%
