# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 16:43:07 2020

@author: dB
"""
import pickle
import pandas as pd
import os
import re
import numpy as np
from shutil import copyfile
   

#master_sweep_dir = 'save_sims_sweep_real_cols'  
#csv_path = 'sweep_sq_dist_sweep_real_cols.csv'
master_sweep_dir = 'save_sims_sensitivity_sweep'
sweep_dirs = os.listdir(master_sweep_dir)
n_sims = len(sweep_dirs)

thermo_hygro_glomeruli = np.array(['VP1d', 'VP1l', 'VP1m', 'VP2', 'VP3', 'VP4', 'VP5'])

all_dfs = []

failed_ones = []

print('scanning through {} sims...'.format(n_sims))

for cur_dir in sweep_dirs:
    
    try:
        # extract sim parameters
        a_e_i_p_str = re.findall('_all(\d+\.?\d*)_ecol(\d+\.?\d*)_icol(\d+\.?\d*)_pcol(\d+\.?\d*)', cur_dir)[0]
        a_e_i_p_vals = [float(x) for x in a_e_i_p_str]

        time_tag = re.findall('([\d\_\-]*)\_\_', cur_dir)[0]
        # load ORN, uPN activities:
        df_AL_activity = pd.read_csv(os.path.join(master_sweep_dir, cur_dir, 'df_AL_activity.csv'), index_col=0)
        sim_params_seed_d = pickle.load( open( os.path.join(master_sweep_dir, cur_dir, 
                                                            'sim_params_seed.p'), "rb" ) )
        odor_names = sim_params_seed_d['odor_panel']
        df_orn_activity = df_AL_activity.loc[df_AL_activity.neur_type == 'ORN'].set_index('neur_name')
        df_orn_frs = df_orn_activity.loc[:, df_orn_activity.columns.str.contains('dur')]
        df_orn_frs.columns = odor_names
        df_upn_activity = df_AL_activity.loc[(df_AL_activity.neur_type == 'uPN') & ~(df_AL_activity.glom.isin(thermo_hygro_glomeruli))].set_index('neur_name')
        df_upn_frs = df_upn_activity.loc[:, df_upn_activity.columns.str.contains('dur')]
        df_upn_frs.columns = odor_names

        all_dfs.append({'simvals': a_e_i_p_vals,
                        'df_AL_activity': df_AL_activity})
        
        
    except:
        print('failed to process'); print(cur_dir)
        failed_ones.append(cur_dir)
        pass
    

pickle.dump(all_dfs, open('sweep_sens_cols.p', 'wb'))


print('done!')
n_fails = len(failed_ones)

if n_fails > 0:
    print('failed to do {} of {}:'.format(n_fails, n_sims))
    print(failed_ones)

copyfile(__file__, os.path.join(master_sweep_dir, 'export_script.py'))