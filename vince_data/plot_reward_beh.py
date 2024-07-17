#%% Import packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
# Set the font size globally
import matplotlib
matplotlib.rcParams.update({'font.size': 14})
import seaborn as sns

#%% Action plot class
class PlotRewardBeh:
    def __init__(self, data):
        self.data = data
        self.numSubs = len(data['subID'].unique())
        self.sublist = data['subID'].unique()
        self.numSess = len(data['sessNo'].unique())
        self.sesslist = data['sessNo'].unique()
        self.subject_palette = sns.color_palette("viridis_r", n_colors=self.numSubs)
        self.session_palette = sns.diverging_palette(145, 300, 
                                                     s=60, as_cmap=False)
                
                
    def within_session_high(self):
    
        # Figure: within-session average hit rate
        numRows = int(np.sqrt(self.numSubs))
        numCols = int(np.ceil(self.numSubs/numRows))
        tick_panel_idx = (numRows - 1) * numCols
        fig1, ax1 = plt.subplots(numRows, numCols,
                                figsize=(numRows, numCols), sharex=False, sharey=True)
        ax1 = ax1.flatten()
        # Create labels
        fig1.text(0.5, 0.05, 'Trial', ha='center', va='center')
        fig1.text(0.03, 0.5, 'Mean Correct Choice', ha='center', va='center', rotation=90)
        # Create figure title
        fig1.suptitle('Average within-session correct\n per subject')
            
        
        for s, sub in enumerate(self.sublist):
            subDf = self.data[self.data['subID'] == sub]
            
            # Average, sem hit rate across sessions 
            sess_high = subDf.groupby(['trialNo']).convHigh.agg(['mean', 'sem'])
                                    
            # Average across all sessions
            sns.lineplot(x=sess_high.index,
                        y=sess_high['mean'],                    
                        color=self.subject_palette[s],                    
                        ax=ax1[s])
            # Plot the SEM as a shaded region
            ax1[s].fill_between(sess_high.index,    
                        sess_high['mean'] - sess_high['sem'],
                        sess_high['mean'] + sess_high['sem'],
                        alpha=0.5,
                        color=self.subject_palette[s])
            ax1[s].set_ylim(0.25, 1)    
            ax1[s].axhline(0.7, color='black', linestyle='--')     
            ax1[s].set_ylabel('')
            ax1[s].set_xlabel('')
            if s < tick_panel_idx: 
                ax1[s].set_xticks([])
        
        # Show average for group
        group_sess_high = self.data.groupby(['trialNo']).convHigh.agg(['mean', 'sem'])
        
        fig2, ax2 = plt.subplots(figsize=(2,3), sharex=False, sharey=True)
        # Create figure title
        fig2.suptitle('Average within-session correct\n across subjects',
                    fontsize=10)
        
        sns.lineplot(x=group_sess_high.index,
                    y=group_sess_high['mean'],
                    color='green',
                    ax=ax2)
        ax2.fill_between(group_sess_high.index,
                    group_sess_high['mean'] - group_sess_high['sem'],
                    group_sess_high['mean'] + group_sess_high['sem'],
                    alpha=0.5,
                    color='green')
        ax2.set_ylim(0.45, 0.71)            
        ax2.axhline(0.7, color='black', linestyle='--')
        ax2.set_ylabel('Mean Correct Choice')
        ax2.set_xlabel('Trial')
        
        # Split by early vs late sessions
        early_data = self.data[self.data['sessNo'] == self.sesslist.min()]
        late_data = self.data[self.data['sessNo'] == self.sesslist.max()]
        
        # Show early vs late for group
        group_early_sess_high = early_data.groupby(['trialNo']).convHigh.agg(['mean', 'sem'])
        group_late_sess_high = late_data.groupby(['trialNo']).convHigh.agg(['mean', 'sem'])
            
        fig3, ax3 = plt.subplots(figsize=(2,3), sharex=False, sharey=True)
        # Create figure title
        fig3.suptitle('Early vs late within-session correct\n across subjects',
                    fontsize=10)
            
        sns.lineplot(x=group_late_sess_high.index,
                    y=group_late_sess_high['mean'],
                    color=self.session_palette[-1],
                    ax=ax3)
        ax3.fill_between(group_late_sess_high.index,
                    group_late_sess_high['mean'] - group_late_sess_high['sem'],
                    group_late_sess_high['mean'] + group_late_sess_high['sem'],
                    alpha=0.5,
                    color=self.session_palette[-1], label='Late')
        sns.lineplot(x=group_early_sess_high.index,
                    y=group_early_sess_high['mean'],
                    color=self.session_palette[0],
                    ax=ax3)
        ax3.fill_between(group_early_sess_high.index,
                    group_early_sess_high['mean'] - group_early_sess_high['sem'],
                    group_early_sess_high['mean'] + group_early_sess_high['sem'],
                    alpha=0.5,
                    color=self.session_palette[0], label='Early')
        ax3.set_ylim(0.35, 0.71)            
        ax3.axhline(0.7, color='black', linestyle='--')
        ax3.set_ylabel('Mean Correct Choice')
        ax3.set_xlabel('Trial')
        ax3.legend(loc='lower right', fontsize=10)
        
        return fig1, fig2, fig3
    
  
                                                
    def across_session_high(self):
    
        # Figure: within-session average hit rate
        numRows = int(np.sqrt(self.numSubs))
        numCols = int(np.ceil(self.numSubs/numRows))
        tick_panel_idx = (numRows - 1) * numCols
        
        # Init figure
        fig1, ax1 = plt.subplots(numRows, numCols,
                                figsize=(numRows, numCols), sharex=False, sharey=True)
        ax1 = ax1.flatten()
        # Create labels
        fig1.text(0.5, 0.05, 'Trial', ha='center', va='center')
        fig1.text(0.03, 0.5, 'Mean Correct Choice', ha='center', va='center', rotation=90)
        # Create figure title
        fig1.suptitle('Average across-session correct\n per subject')
            
        
        for s, sub in enumerate(self.sublist):
            subDf = self.data[self.data['subID'] == sub]
            
            # Average, sem hit rate across sessions 
            sess_high = subDf.groupby(['sessNo']).highChosen.agg(['mean', 'sem'])
                                    
            # Plot across sessions
            sns.barplot(x=sess_high.index,
                        y=sess_high['mean'],                    
                        color=self.subject_palette[s],                    
                        ax=ax1[s])            
            ax1[s].errorbar(range(len(sess_high)),
                            sess_high['mean'], 
                            yerr=sess_high['sem'])

            ax1[s].set_ylim(0.1, 1)    
            ax1[s].axhline(0.7, color='black', linestyle='--')     
            ax1[s].set_ylabel('')
            ax1[s].set_xlabel('')
            if s < tick_panel_idx: 
                ax1[s].set_xticks([])
            else:
                ax1[s].set_xticks(sess_high.index[::2])
        
        # Show average for group
        group_sess_high = self.data.groupby(['sessNo']).highChosen.agg(['mean', 'sem'])
        
        fig2, ax2 = plt.subplots(figsize=(2,3), sharex=False, sharey=True)
        # Create figure title
        fig2.suptitle('Average across-session correct\n across subjects',
                    fontsize=10)
        
        # Plot across sessions            
        sns.barplot(x=group_sess_high.index,
                    y=group_sess_high['mean'],
                    color='green',
                    ax=ax2)
        ax2.errorbar(range(len(group_sess_high)),
                        group_sess_high['mean'], 
                        yerr=group_sess_high['sem'])

        ax2.set_ylim(0.45, 0.71)            
        ax2.axhline(0.7, color='black', linestyle='--')
        ax2.set_ylabel('Mean Correct Choice')
        ax2.set_xlabel('Trial')
                              
        return fig1, fig2
        
    def hitHist_outHist_high(self):             
        fig, ax = plt.subplots(figsize=(2,3))
        
        # plot two-way interaction between convHit and convOut in predicting highChosen
        sns.barplot(x = self.data['payOut'],
                    y = self.data['highChosen'],
                    hue = self.data['isHit'],
                    palette = ['teal','purple'], alpha=0.5,
                    linewidth=1, edgecolor='black',
                    ci=95,
                    ax=ax)
      
        # Remove legend
        legend_labels = ["No Hit", "Hit"]
        # Change legend labels
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, legend_labels, title="A-Out",
                  loc='upper left')
        
        ax.axhline(0.5, color='black', linestyle='--')
        ax.set_ylim([0.4, 0.90])
        ax.set_xticklabels(['No Win','Win'])
        ax.set_xlabel('R-Out')
        ax.set_ylabel('Proportion correct')
        ax.yaxis.set_major_locator(MultipleLocator(0.2))

        return fig
    