#%% Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymer4.models as pymer


#%% Action plot class
class StatsRewardBeh:
    def __init__(self, data):
        self.data = data.dropna(subset=['payOut'])
        self.numSubs = len(data['subID'].unique())
        self.sublist = data['subID'].unique()
        self.subject_palette = sns.color_palette("viridis_r", n_colors=self.numSubs)
        self.session_palette = sns.color_palette("Greens_r", n_colors=8)
        self.default_colour = 'black'
        self.control_options = "optimizer='Nelder_Mead', optCtrl = list(FtolAbs=1e-8, XtolRel=1e-8)"
    
    def plot_model_coefs(self, IV, grouping, coefficients, estimates): 
        est, upper, lower = estimates.loc[IV].values
                                        

        # Plot effect (estimates and coefficients)
        fig, ax = plt.subplots(figsize=(2,3))
        sns.stripplot(data=coefficients, x=0.1, y=IV, 
                      jitter=0.04, hue = grouping,
                      palette = self.session_palette,
                      ax=ax)
        ax.legend_.remove()
        
        # Draw point range plot
        ax.vlines(-0.1, lower, upper, color=self.default_colour, linestyle='-', lw=2)    
        ax.scatter(-0.1, est, s=50, c=self.default_colour, zorder=3)    
        ax.axhline(0, c='r', linestyle='--')
        
        # Draw violin plot
        parts = ax.violinplot(coefficients[IV],
                            positions = [0.1],
                            showmeans=False, showmedians=False, 
                            showextrema=False)
        
        for pc in parts['bodies']:
            pc.set_facecolor(self.default_colour)
            pc.set_edgecolor(self.default_colour)
            pc.set_alpha(0.3)    
            # get the center
            m = np.mean(pc.get_paths()[0].vertices[:, 0])
            # modify the paths to not go further left than the center
            pc.get_paths()[0].vertices[:, 0] = np.clip(pc.get_paths()[0].vertices[:, 0], m, np.inf)        
            
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
        ax.set_xticks([])
        ax.set_ylabel('Coefficient')
                
        return fig
            
    def get_model_coefs(self, model, grouping):
        # Get estimates
        res = model.coefs
        estimate_Df = res[['Estimate', '2.5_ci', '97.5_ci']]
        
        # Create model of coefficients         
        coef_Df = model.fixef[0] if isinstance(model.fixef, list) else model.fixef
        coef_Df.reset_index(inplace=True)
        coef_Df[grouping] = coef_Df['index'].str.split(':', expand=True)

        return coef_Df, estimate_Df, res
                
    def trial_predict_high(self):
        # Mean-center the predictor 
        grouped_data = self.data.groupby(['subID', 'sessNo'])
        self.data['trialNo_mc'] = grouped_data['trialNo'].transform(lambda x: x - x.mean())
                
        # Fit model 
        group_model = pymer.Lmer('highChosen ~ trialNo_mc + (-1 + trialNo_mc | subID / sessNo)',
                                 family='binomial', data=self.data)
        group_model.fit(control=self.control_options)    
        
        # Get coefficients
        grouping_factors = ['session','subject']     
        coef_Df, estimate_Df, group_res = self.get_model_coefs(group_model, grouping_factors)
        
        # Plot model
        group_fig = self.plot_model_coefs('trialNo_mc', grouping_factors[0], coef_Df, estimate_Df)
        
        return group_res, group_fig
        

    
    def hit_out_interact_high(self):                
        # Fit model 
        group_model = pymer.Lmer('highChosen ~ isHit:payOut + (1 +  isHit:payOut | subID / sessNo )',
                                 family='binomial', data=self.data)
        group_model.fit(control=self.control_options)    
        
        # Get coefficients
        grouping_factors = ['session','subject']     
        coef_Df, estimate_Df, group_res = self.get_model_coefs(group_model, grouping_factors)
        
        # Plot model
        group_fig_interact = self.plot_model_coefs('isHit:payOut', grouping_factors[0], coef_Df, estimate_Df)
        
        return group_res, group_fig_interact
    
    def create_autoreg_formula(self, DV, IV, n_back, grouping):
        formula = ""
        if isinstance(IV, list):
            for indep_var in IV:
                for n in range(1, n_back+1):
                    formula += f"{indep_var}_{n}_back + "
            formula = formula.rstrip(" + ")
        else: 
            for n in range(1, n_back+1):
                formula += f"{IV}_{n}_back + "
            formula = formula.rstrip(" + ")
        if len(grouping) == 2:
            final_string = f"{DV} ~ {formula} + (1 | {grouping[1]} / {grouping[0]})"
        else: 
            final_string = f"{DV} ~ {formula} + (1 | {grouping[0]})"
        return final_string
    
            
    def plot_autoreg(self, estimates, n_back=10): 
        # Create a figure and axis
        parameter = estimates['Estimate']
        lowers = parameter - estimates['2.5_ci']
        uppers = estimates['97.5_ci'] - parameter

        # Remove intercept
        parameter, lowers, uppers = parameter[1:], lowers[1:], uppers[1:]
        # Create x-axis
        n_backs = np.arange(n_back)+1  # Generate array from 10 to 1

        fig, ax = plt.subplots(figsize=(2,2))
        ax.scatter(x=n_backs,
                y=parameter,
                c=self.default_colour,)

        ax.errorbar(x = n_backs,
                    y = parameter,
                    yerr = (lowers,uppers),
                    linestyle = '-',
                    c=self.default_colour,)
        ax.invert_xaxis()

        ax.axhline(0, color='red', linestyle='--')
        ax.set_ylabel('Coefficient')
        ax.set_xlabel('n-back Trials')

        return fig
    
       
    def past_hit_high(self):
        grouping_factors = ['subID']     
        
        # Fit model 
        group_model = pymer.Lmer(self.create_autoreg_formula(DV='highChosen', IV='isHit', 
                                                             n_back = 10,
                                                             grouping = grouping_factors),
                                 family='binomial', data=self.data)
        group_model.fit(control=self.control_options)    
        
        # Get coefficients
        _, estimate_Df, group_res = self.get_model_coefs(group_model, grouping_factors)
        
        # Plot model
        group_fig = self.plot_autoreg(estimate_Df)
        
        return group_res, group_fig
    
    def past_out_high(self):
        grouping_factors = ['subID']     
        
        # Fit model 
        group_model = pymer.Lmer(self.create_autoreg_formula(DV='highChosen', IV='outcome', 
                                                             n_back = 10,
                                                             grouping = grouping_factors),
                                 family='binomial', data=self.data)
        group_model.fit(control=self.control_options)    
        
        # Get coefficients
        _, estimate_Df, group_res = self.get_model_coefs(group_model, grouping_factors)
        
        # Plot model
        group_fig = self.plot_autoreg(estimate_Df)
        
        return group_res, group_fig
    
    
    
    
    def plot_autoreg_multivar(self, estimates, vars, n_back=10): 
               
        # Create x-axis
        n_backs = np.arange(n_back)+1  # Generate array from 10 to 1
        colors = ['green', 'red']
        labels = ['ROut','AOut']
        fig, ax = plt.subplots(figsize=(2,2))
       
        # Split estimates by the variables
        for v, var in enumerate(vars): 
            var_estimates =  estimates[estimates.index.str.contains(var)]
                    
            # Create a figure and axis
            parameter = var_estimates['Estimate']
            lowers = parameter - var_estimates['2.5_ci']
            uppers = var_estimates['97.5_ci'] - parameter


            ax.scatter(x=n_backs,
                    y=parameter[:10],
                    c=colors[v],
                    label=labels[v])

            ax.errorbar(x = n_backs,
                        y = parameter[:10],
                        yerr = (lowers[:10],uppers[:10]),
                        linestyle = '-',
                        c=colors[v])            
        
            
        ax.invert_xaxis()
        ax.legend()
        ax.axhline(0, color='black', linestyle='--')
        ax.set_ylabel('Coefficient')
        ax.set_xlabel('n-back Trials')

        return fig
    
        
    def create_conditioned_df(self, n_back=10):
        # Create columnns for whether the current response is left or right
        self.data['isLeft'] = np.array(1 - self.data['responseKey'], dtype=bool)
        self.data['isRight'] = self.data['responseKey'].astype(bool)
        
        # Subset data to only where stim1 is on the left and right        
        stim1_left = self.data[self.data['stim1_left'] == True]        
        stim1_right = self.data[self.data['stim1_left'] == False]
                
        # Create a column for outcomes conditioned on stim1
        stim1_left['outcome_conditioned_s1'] = stim1_left['payOut'].multiply(stim1_left['selected_stim1'], axis=0)
        stim1_left['outcome_conditioned_s2'] = stim1_left['payOut'].multiply(stim1_left['selected_stim2'], axis=0)        
        # Create a column for actions conditioned on side 
        stim1_left['hit_conditioned'] = stim1_left['isHit'].multiply(stim1_left['isLeft'], axis=0)
        
        # Create a column for outcomes conditioned on stim1
        stim1_right['outcome_conditioned_s1'] = stim1_right['payOut'].multiply(stim1_right['selected_stim1'], axis=0)
        stim1_right['outcome_conditioned_s2'] = stim1_right['payOut'].multiply(stim1_right['selected_stim2'], axis=0)        
        # Create a column for actiosn conditioend on side
        stim1_right['hit_conditioned'] = stim1_right['isHit'].multiply(stim1_right['isRight'], axis=0)
        
        # Create new concatenated dataframe 
        cols_to_keep = ['subID', 'sessNo', 'trialNo', 'selected_stim1', 
                        'outcome_conditioned_s1', 'outcome_conditioned_s2', 'hit_conditioned']
        conditioned_Df = pd.concat([stim1_left[cols_to_keep], stim1_right[cols_to_keep]])
        
        
        # Get n-back hit and high choice
        for n in range(1, n_back+1):
            conditioned_Df.loc[:, f'outcome_conditioned_s1_{n}_back'] = conditioned_Df['outcome_conditioned_s1'].shift(n)
            conditioned_Df.loc[:, f'outcome_conditioned_s2_{n}_back'] = conditioned_Df['outcome_conditioned_s2'].shift(n)
            conditioned_Df.loc[:, f'hit_conditioned_{n}_back'] = conditioned_Df['hit_conditioned'].shift(n)
            
        return conditioned_Df
            
    def hit_versus_outcome(self): 
        
        conditioned_Df = self.create_conditioned_df()
        
        grouping_factors = ['subID']
        # All trials and sessions
        # Fit model conditioned on selecting stim 1
        model_outcome_versus_hit = pymer.Lmer(self.create_autoreg_formula(DV='selected_stim1', IV=['outcome_conditioned_s1','hit_conditioned'], 
                                                             n_back = 10,
                                                             grouping = grouping_factors),
                                 family='binomial', data=conditioned_Df)
        model_outcome_versus_hit.fit(control=self.control_options)                    
        # Get coefficients
        _, coef_outcome_versus_hit, _ = self.get_model_coefs(model_outcome_versus_hit, grouping_factors)        
        # Plot model
        group_fig = self.plot_autoreg_multivar(coef_outcome_versus_hit, vars=['outcome','hit'])
        
        return     
      
    def plot_conditioned_autoreg(self, conditions, labels): 
       
        fig, ax = plt.subplots(figsize=(2,2))
        
        
        # iterate over conditions and labels
        for condition, cond_label in zip(conditions, labels):
            parameter = condition['Estimate']
            lowers = parameter - condition['2.5_ci']
            uppers = condition['97.5_ci'] - parameter
            
            # Create x-axis
            n_backs = np.arange(len(condition)) + 1

            # Plot condition 1
            ax.scatter(x=n_backs,
                    y=parameter,
                    label=cond_label)
            ax.errorbar(x = n_backs,
                        y = parameter,
                        yerr = (lowers, uppers),
                        linestyle = '-')
        
        ax.invert_xaxis()
        ax.axhline(0, color='red', linestyle='--')
        ax.set_ylabel('p(stay)')
        ax.set_xlabel('n-back Trials')
        ax.legend(fontsize=10)
        
        return fig
    
        
    def conditioned_outcome_stim(self, n_back=10):
        
        # Get n-back hit and high choice
        for n in range(1,n_back+1):
            # Create contrast coding of selected stim1 (+1) and selected stim2 (-1)
            self.data[f'selected_stim_{n}_back'] = self.data[f'selected_stim1_{n}_back'] - self.data[f'selected_stim2_{n}_back']        
            # Create vectors encoding selected stim on win trials
            self.data[f'selected_stim_win_{n}_back'] = self.data[f'win_{n}_back'].multiply(self.data[f'selected_stim_{n}_back'], axis=0)
            # Create vectors encoding selected stim on loss trials, split by reason for loss
            self.data[f'selected_stim_loss_hit_{n}_back'] = self.data[f'loss_hit_{n}_back'].multiply(self.data[f'selected_stim_{n}_back'], axis=0)
            self.data[f'selected_stim_loss_miss_{n}_back'] = self.data[f'loss_miss_{n}_back'].multiply(self.data[f'selected_stim_{n}_back'], axis=0)
            
            
        grouping_factors = ['subID']                     
        # Fit model conditioned on selecting stim 2
        model_cond_outcome = pymer.Lmer(self.create_autoreg_formula(DV='selected_stim1', IV=['selected_stim_win',
                                                                                             'selected_stim_loss_hit', 
                                                                                             'selected_stim_loss_miss'],
                                                             n_back = 10,
                                                             grouping = grouping_factors),
                                 family='binomial', data=self.data)
        model_cond_outcome.fit(control=self.control_options)                    
        # Get coefficients
        _, coef_cond_outcome, _ = self.get_model_coefs(model_cond_outcome, grouping_factors)
        
        # Split coefficients by condition
        coef_cond_win = coef_cond_outcome[coef_cond_outcome.index.str.contains('win')]
        coef_cond_loss_hit = coef_cond_outcome[coef_cond_outcome.index.str.contains('loss_hit')]
        coef_cond_loss_miss = coef_cond_outcome[coef_cond_outcome.index.str.contains('loss_miss')]
       
        # Plot model
        group_fig = self.plot_conditioned_autoreg([coef_cond_win, coef_cond_loss_hit, coef_cond_loss_miss],
                                                  ['win','loss','miss'])
        
        return group_fig
    
    

    
    def conditioned_hit_correct_stim(self, n_back = 10):
        
        # Get n-back hit and high choice
        for n in range(1,n_back+1):
            # Create contrast coding of selected stim1 (+1) and selected stim2 (-1)
            self.data[f'selected_stim_{n}_back'] = self.data[f'selected_stim1_{n}_back'] - self.data[f'selected_stim2_{n}_back']        
            # Create vectors encoding selected stim on win trials
            self.data[f'selected_stim_hit_{n}_back'] = self.data[f'isHit_{n}_back'].multiply(self.data[f'selected_stim_{n}_back'], axis=0)
            
           
        grouping_factors = ['subID']             
      
        # Fit model conditioned on selecting stim 1 when stim 1 is correct
        model_cond_act_correct = pymer.Lmer(self.create_autoreg_formula(DV='selected_stim1', IV='selected_stim_hit',
                                                             n_back = 10,
                                                             grouping = grouping_factors),
                                 family='binomial', data=self.data[self.data['stim1_High'] == True])
        model_cond_act_correct.fit(control=self.control_options)                    
        # Get coefficients
        _, coef_cond_act_correct, _ = self.get_model_coefs(model_cond_act_correct, grouping_factors)
        
        # Fit model conditioned on selecting stim 1 when stim 1 is incorrect
        model_cond_act_incorrect = pymer.Lmer(self.create_autoreg_formula(DV='selected_stim1', IV='selected_stim_hit',
                                                             n_back = 10,
                                                             grouping = grouping_factors),
                                 family='binomial', data=self.data[self.data['stim1_High'] == False])
        model_cond_act_incorrect.fit(control=self.control_options)                    
        # Get coefficients
        _, coef_cond_act_incorrect, _ = self.get_model_coefs(model_cond_act_incorrect, grouping_factors)
        
        
        # Split coefficients by condition
        coef_cond_hit_correct = coef_cond_act_correct[coef_cond_act_correct.index.str.contains('hit')]        
        coef_cond_hit_incorrect = coef_cond_act_incorrect[coef_cond_act_incorrect.index.str.contains('hit')]
        
        
                    
        # Plot model
        group_fig = self.plot_conditioned_autoreg([coef_cond_hit_correct, coef_cond_hit_incorrect],
                                                         ['correct','incorrect'])
        
        return group_fig
    
    
    