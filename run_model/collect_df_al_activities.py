# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 16:43:07 2020

@author: dB
"""
import pickle
import pandas as pd
from datetime import datetime
import os

# get timetag
n = datetime.now()
time_tag = '{}_{}_{}-{}_{}_{}'.format(\
      n.year, n.month, n.day, n.hour, n.minute, n.second)

master_sweep_dir = 'save_sims_synapticNoise_ORNs_LNs_PNs'
save_to_path = os.path.join(master_sweep_dir, 'd_sims_res_df_AL_activities_{}.p'.format(time_tag))


sweep_dirs = os.listdir(master_sweep_dir)
n_sims = len(sweep_dirs)

failed_ones = []

print('scanning through {} sims...'.format(n_sims))

d_sims = {}

for cur_dir in sweep_dirs:
    
    print('scanning {}'.format(cur_dir))
    
    d_sims[cur_dir] = {'success': 0}
    nfail = 0
    # try saving model outputs
    try:
        df_AL_activity = pd.read_csv(os.path.join(master_sweep_dir, cur_dir, 'df_AL_activity.csv'), index_col=0)
        d_sims[cur_dir]['df_AL_activity'] = df_AL_activity
    except:
        nfail += 1
        print('\tno df_AL_activity')
    try:
        df_neur_ids = pd.read_csv( os.path.join(master_sweep_dir, cur_dir, 'df_neur_ids.csv'), index_col=0)
        d_sims[cur_dir]['df_neur_ids'] = df_neur_ids
    except:
        nfail += 1
        print('\tno df_neur_ids')
    try:
        ln_resample_set = pd.read_csv( os.path.join(master_sweep_dir, cur_dir, 'LN_resample_set.csv'), index_col=0)
        d_sims[cur_dir]['ln_resample_set'] = ln_resample_set
    except:
        print('\tno LN_resample_set')
    if nfail == 0:
        d_sims[cur_dir]['success'] = 1
    elif nfail == 2:        
        print('\tfailed to process')
        failed_ones.append(cur_dir)
# save
pickle.dump(d_sims, open(save_to_path, 'wb'))

print('done!')

# process fails
n_fails = len(failed_ones)
if n_fails > 0:
    print('failed to do {} of {}:'.format(n_fails, n_sims))
    print(failed_ones)