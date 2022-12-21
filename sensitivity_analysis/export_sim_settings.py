#!/usr/bin/env python
# coding: utf-8

print('loading packages...')

import os
import sys 

# set home directory (the path to ALModel_dev/)
file_path = os.path.abspath(__file__)
project_dir = os.path.join(file_path.split('ALModel_dev')[0], 'ALModel_dev')
if not os.path.exists(project_dir):
    raise NameError('set path to ALModel_dev')
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

args = parser.parse_args()

MULT_ALL = args.mA; MULT_ORN = args.mO; MULT_ELN = args.mE; MULT_ILN = args.mI; MULT_PN = args.mP

def draw_log_noise_param_uniform(a, max_width=0.25):
    '''
    Given a parameter (for instance, PN sensitivity = 0.4),
    and a desired max width (for instance, 0.25:
    lower: 0.4 * (1 - 0.25) = 0.4 * 3/4,
    upper: 0.4 * 4/3),
    drawns a random value spaced uniformly in log space
    of the parameter +/- max_width
    '''
    size_log_scale_under = -np.log(1-max_width)
    lb_log = np.log(a) - size_log_scale_under
    ub_log = np.log(a) + size_log_scale_under
    drawn_a_log = np.random.uniform(lb_log, ub_log)
    drawn_a = np.exp(drawn_a_log)
    return drawn_a

def draw_log_noise_param_norm(a, sd=0.25):
    '''
    Given a parameter (for instance, PN sensitivity = 0.4),
    and a desired standard deviation (for instance, 0.25:
    lower: 0.4 * (1 - 0.25) = 0.4 * 3/4,
    upper: 0.4 * 4/3),
    drawns a random value spaced normally in log space
    centered at parameter with desired sd
    '''
    sd_log_scale_under = -np.log(1-sd)
    drawn_a_log = np.random.normal(loc=np.log(a), scale=sd_log_scale_under)
    drawn_a = np.exp(drawn_a_log)
    return drawn_a

MULT_ORN = draw_log_noise_param_norm(MULT_ORN, sd=0.5)
MULT_ELN = draw_log_noise_param_norm(MULT_ELN, sd=0.5)
MULT_ILN = draw_log_noise_param_norm(MULT_ILN, sd=0.5)
MULT_PN = draw_log_noise_param_norm(MULT_PN, sd=0.5)

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
                        '1-butanol'
                        ])
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

run_tag = f'0v12_all{MULT_ALL}_ocol{col_orn}_ecol{col_eln}_icol{col_iln}_pcol{col_pn}_sensitivity_MAC_odors_{sec_tag}'
run_explanation = '''
v1.2 of hemibrain, with ORNs/LNs/uPNs/mPNs
using ALS imputed Bhandawat 2007 odors
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