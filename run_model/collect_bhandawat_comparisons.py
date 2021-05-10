# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 16:43:07 2020

@author: dB
"""

import pandas as pd
import os
import sys
import re
import numpy as np
from shutil import copyfile

file_path = os.path.abspath(__file__)
project_dir = os.path.join(file_path.split('ALVariability')[0], 'ALVariability')
sys.path.append(project_dir)
from utils.analysis_utils import make_comparison_plots, make_comparison_pdf

#master_sweep_dir = 'save_sims_sweep_real_cols'  
#csv_path = 'sweep_sq_dist_sweep_real_cols.csv'
master_sweep_dir = 'C:/Users/dB/deBivort/projects/ALVariability/run_model/save_sims_sensitivity_sweep'
sweep_dirs = os.listdir(master_sweep_dir)
n_sims = len(sweep_dirs)

thermo_hygro_glomeruli = np.array(['VP1d', 'VP1l', 'VP1m', 'VP2', 'VP3', 'VP4', 'VP5'])
failed_ones = []

print('scanning through {} sims...'.format(n_sims))

all_pdfs_folder = 'save_pdfs_here'
if not os.path.exists(all_pdfs_folder):
    os.makedirs(all_pdfs_folder)

for cur_dir in sweep_dirs:
    
    try:
        # extract sim parameters
        a_e_i_p_str = re.findall('_all(\d+\.?\d*)_ecol(\d+\.?\d*)_icol(\d+\.?\d*)_pcol(\d+\.?\d*)', cur_dir)[0]
        
        sim_tag = '_a{}_e{}_i{}_p{}'.format(*a_e_i_p_str)

        time_tag = re.findall('([\d\_\-]*)\_\_', cur_dir)[0]
        # load ORN, uPN activities:
        df_AL_activity = pd.read_csv(os.path.join(master_sweep_dir, cur_dir, 'df_AL_activity.csv'), index_col=0)
        
        bhandawat_comparison_plots_dir = os.path.join(master_sweep_dir, cur_dir, 'bhandawat_comparison_plots')
        if not os.path.exists(bhandawat_comparison_plots_dir):
            os.makedirs(bhandawat_comparison_plots_dir)
          
               
        make_comparison_plots(df_AL_activity, bhandawat_comparison_plots_dir)
        pdf_loc = make_comparison_pdf(bhandawat_comparison_plots_dir, msg=sim_tag)
        
        new_fig_name = os.path.join(all_pdfs_folder, 'bhandawat_compare_sim_' + sim_tag + '.pdf')
    
        copyfile(pdf_loc, new_fig_name)
        
        
    except:
        print('failed to process'); print(cur_dir)
        failed_ones.append(cur_dir)
        pass
    

print('done!')
n_fails = len(failed_ones)

if n_fails > 0:
    print('failed to do {} of {}:'.format(n_fails, n_sims))
    print(failed_ones)

copyfile(__file__, os.path.join(master_sweep_dir, 'export_script_comp_bhandawat.py'))