#!/usr/bin/env python
# coding: utf-8

print('loading packages...')

import os
import sys 

# set home directory (the path to ALVariability/)
file_path = os.path.abspath(__file__)
project_dir = os.path.join(file_path.split('ALVariability')[0], 'ALVariability')
if not os.path.exists(project_dir):
    raise NameError('set path to ALVariability')
sys.path.append(project_dir)


import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from utils.model_params import params as hemi_params
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mA', type=float, default=0.1, help='multiplier on all columns')
parser.add_argument('--mO', type=float, default=1, help='multiplier on ORN column')
parser.add_argument('--mE', type=float, default=0.4, help='multiplier on eLN column')
parser.add_argument('--mI', type=float, default=0.2, help='multiplier on iLN column')
parser.add_argument('--mP', type=float, default=4, help='multiplier on PN column')
parser.add_argument('--rO', type=int, default=0, help='binary, resample ORNs?')
parser.add_argument('--rL', type=int, default=0, help='binary, resample LNs?')
parser.add_argument('--rP', type=int, default=0, help='binary, resample PNs?')
parser.add_argument('--ruP', type=int, default=1, help='binary, resample uPNs? only active if --rP on')
parser.add_argument('--rmP', type=int, default=1, help='binary, resample mPNs? only active if --rP on')

args = parser.parse_args()

MULT_ALL = args.mA; MULT_ORN = args.mO; MULT_ELN = args.mE; MULT_ILN = args.mI; MULT_PN = args.mP
RESAMPLE_ORNs = args.rO; RESAMPLE_LNs = args.rL; RESAMPLE_PNs = args.rP; RESAMPLE_uPNs = args.ruP; RESAMPLE_mPNs = args.rmP

print('setting settings...')

# set master directory
master_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'save_sims_sensitivity_sweep')
if not os.path.exists(master_save_dir):
    os.mkdir(master_save_dir)
    
# set time tag
n = datetime.now()
day_tag = '{}_{}_{}'.format(n.year, n.month, n.day)
sec_tag = '{}_{}_{}'.format(n.hour, n.minute, n.second)
time_tag = day_tag + '-' + sec_tag
     
#### SET PARAMETERS
    
# odors, sequentially
mac_odors = np.array(['3-octanol',
                        '1-hexanol',
                        'ethyl lactate',
                        #'citronella',
                        '2-heptanone',
                        '1-pentanol',
                        'ethanol',
                        'geranyl acetate',
                        'hexyl acetate',
                        '4-methylcyclohexanol',
                        'pentyl acetate',
                        '1-butanol'])
odor_panel = mac_odors

# duration of odor stimulus (seconds)
odor_dur = 0.4

# break between odor stimuli (seconds)
odor_pause = 0.25

# simulation time after last stimulus (seconds)
end_padding = 0.01


# load hemibrain + odor imputation data
projects_path = os.path.join(file_path.split('projects')[0], 'projects')
df_neur_ids = pd.read_csv(os.path.join(project_dir, 'connectomics/hemibrain_v1_2/df_neur_ids.csv'), index_col=0)
al_block = pd.read_csv(os.path.join(project_dir, 'connectomics/hemibrain_v1_2/AL_block.csv'), index_col=0)
imput_table = pd.read_csv(os.path.join(project_dir, 'odor_imputation/df_odor_door_all_odors_imput_ALS.csv'), index_col=0)


# set excitatory LN positions
np.random.seed(1234)
LN_bodyIds = df_neur_ids[df_neur_ids.altype == 'LN'].bodyId.values
num_LNs = len(LN_bodyIds)
num_eLNs = int(np.round(num_LNs / 6.4))
topk = int(np.round(num_LNs / 2))
nlns = len(LN_bodyIds)
elnpos = np.random.choice(np.arange(topk), num_eLNs, replace=False) 
print(elnpos)

#########
######### RESAMPLING
#########

df_neur_ORNs = df_neur_ids.copy()[df_neur_ids.altype == 'ORN']
df_neur_LNs = df_neur_ids.copy()[df_neur_ids.altype == 'LN']
df_neur_PNs = df_neur_ids.copy()[df_neur_ids.altype.isin(['uPN', 'mPN'])]

final_ORN_ids = df_neur_ORNs.bodyId.values
final_LN_ids = df_neur_LNs.bodyId.values
final_PN_ids = df_neur_PNs.bodyId.values

if RESAMPLE_ORNs:
    random_ORN_sample = []
    orn_gloms = df_neur_ORNs.glom.unique() 
    for g in orn_gloms:
        glom_orn_bodyIds = df_neur_ORNs[df_neur_ORNs.glom == g].bodyId.values
        random_glom_ORN_sample = np.random.choice(glom_orn_bodyIds, len(glom_orn_bodyIds), replace=True)
        random_ORN_sample.append(random_glom_ORN_sample)
    random_ORN_sample = np.concatenate(random_ORN_sample)
    final_ORN_ids = random_ORN_sample

if RESAMPLE_LNs:
    random_LN_sample = np.random.choice(final_LN_ids, len(final_LN_ids), replace=True)
    df_neur_LNs.glom = pd.to_numeric(df_neur_LNs.glom)
    random_LN_sample_sorted_by_charId = (df_neur_LNs
        .set_index('bodyId')
        .loc[random_LN_sample]
        .sort_values('glom', ascending=True)
        ).index.values
    final_LN_ids = random_LN_sample_sorted_by_charId

if RESAMPLE_PNs:
    upn_bodyIds = df_neur_PNs[df_neur_PNs.altype == 'uPN'].bodyId.values
    final_uPN_sample = upn_bodyIds
    if RESAMPLE_uPNs:
        # resample within PN glomeruli to get random uPN sample
        random_uPN_sample = []
        pn_gloms = df_neur_PNs.glom.unique() 
        for g in pn_gloms:
            glom_pn_bodyIds = df_neur_PNs[df_neur_PNs.glom == g].bodyId.values
            random_glom_PN_sample = np.random.choice(glom_pn_bodyIds, len(glom_pn_bodyIds), replace=True)
            random_uPN_sample.append(random_glom_PN_sample)
        final_uPN_sample = np.concatenate(random_uPN_sample)
    
    mpn_bodyIds = df_neur_PNs[df_neur_PNs.altype == 'mPN'].bodyId.values
    final_mPN_sample = mpn_bodyIds
    if RESAMPLE_mPNs:
        # resample the multi-PNs
        final_mPN_sample = np.random.choice(mpn_bodyIds, len(mpn_bodyIds), replace=True)
    # concatenate
    final_PN_ids = np.concatenate((final_uPN_sample, final_mPN_sample))

# get final resampled df_char_ids
final_bodyIds = np.concatenate((final_ORN_ids, final_LN_ids, final_PN_ids))
df_neur_ids_resampled = df_neur_ids.set_index('bodyId').loc[final_bodyIds].reset_index()[df_neur_ids.columns]

# and, reorder al_block
al_block.columns = al_block.columns.astype(np.int64)
al_block_resampled = al_block.loc[final_bodyIds, final_bodyIds]

resamp_tag = '{}{}{}'.format('ORN_'*RESAMPLE_ORNs, 
                            'LN_'*RESAMPLE_LNs, 
                            '{}{}PN_'.format('u'*RESAMPLE_uPNs,
                                             'm'*RESAMPLE_mPNs)*RESAMPLE_PNs)

#########
######### RESAMPLING
#########

# set ORN decay
decay_tc = 0.11 # seconds
decay_fadapt = 0.75 # a fraction

col_orn = MULT_ORN
col_iln = MULT_ILN
col_eln = MULT_ELN
col_pn = MULT_PN

custom_scale_dic = {
    'ALL': MULT_ALL,
    'otoo': col_orn, 'otoi': col_iln, 'otoe': col_eln, 'otop': col_pn,
    'itoo': col_orn, 'itoi': col_iln, 'itoe': col_eln, 'itop': col_pn,
    'etoo': col_orn, 'etoi': col_iln, 'etoe': col_eln, 'etop': col_pn,
    'ptoo': col_orn, 'ptoi': col_iln, 'ptoe': col_eln, 'ptop': col_pn,
   }

hemi_params['odor_rate_max'] = 400

run_tag = f'0v12_all{MULT_ALL}_ecol{col_eln}_icol{col_iln}_pcol{col_pn}_resample_{resamp_tag}_{sec_tag}'
run_explanation = '''
v1.2 of hemibrain, with ORNs/LNs/uPNs/mPNs
using ALS imputed MAC odors
all x0.1, eLNs x0.4, iLNs x0.2, PNs x4
1/6.4 of LNs are set as excitatory (Tsai et al, 2018), 
    drawn randomly from top 50\% of LNs when sorted by number of glomeruli innervated 
ORN decay timescale 110 ms to 75% (Kao and Lo, 2020)
'''

# erase output
erase_sim_output = 1

##### Set export directory
saveto_dir = os.path.join(master_save_dir, time_tag+'__'+run_tag)
if not os.path.exists(saveto_dir):
    os.mkdir(saveto_dir)
    
##### SAVE INFO
print('saving sim_params_seed.p...')
sim_params_seed = {
    'odor_panel': odor_panel,
    'odor_dur': odor_dur,
    'odor_pause': odor_pause, 
    'end_padding': end_padding,
    'project_dir': project_dir,
    'hemi_params': hemi_params,
    'elnpos': elnpos,
    'custom_scale_dic': custom_scale_dic,
    'run_tag': run_tag,
    'run_explanation': run_explanation,
    'decay_tc': decay_tc,
    'decay_fadapt': decay_fadapt,
    'erase_sim_output': erase_sim_output,
    'imputed_glom_odor_table': imput_table,
    'df_neur_ids': df_neur_ids,
    'al_block': al_block
    }

pickle.dump(sim_params_seed,
            open(os.path.join(saveto_dir, 'sim_params_seed.p'), "wb" ))

print('saving other files...')

# write run_sim.py
with open('run_sim_template.py', 'r') as rf:
    template = rf.readlines()
    with open(os.path.join(saveto_dir, 'run_sim.py'), 'w') as wf:
        wf.writelines(template)
        
# write cluster submission script
with open('submit_to_cluster_template.sh', 'r') as rf:
    cluster_template = rf.readlines()
    with open(os.path.join(saveto_dir, 'submit_job.sh'), 'w') as wf:
        wf.writelines(cluster_template)
        
# write notes
with open(os.path.join(saveto_dir, 'run_notes.txt'), 'w') as f:
    f.write(run_explanation)
        
      
# write this file
with open(os.path.abspath(__file__), 'r') as rf:
    this_file = rf.readlines()
    with open(os.path.join(saveto_dir, 'export_settings_copy.py'), 'w') as wf:
        wf.writelines(this_file)
        
# write the saveto directory
with open('cur_saveto_dir.txt', 'w') as f:
    f.write(saveto_dir)

print('done')