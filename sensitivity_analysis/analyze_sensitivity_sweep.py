import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

import os
import sys

file_path = os.path.abspath(__file__)
project_dir = os.path.join(file_path.split('ALModel_dev')[0], 'ALModel_dev')

sensitivity_analysis_dir = os.path.join(project_dir, 'sensitivity_analysis')
saved_sims_dir = os.path.join(sensitivity_analysis_dir, 'save_sims_sensitivity_sweep')

## map simulation time tags to run condition

base_timetag = {'2022_11_9-11_33_0': 'base'}

d_0p25_timetags = {
	'2022_11_9-11_34_16': 'ORN x3/4',
	'2022_11_9-11_34_28': 'eLN x3/4',
	'2022_11_9-11_35_9': 'iLN x3/4',
	'2022_11_9-11_35_16': 'PN x3/4',

	'2022_11_10-12_24_10': 'ORN x4/3',
	'2022_11_9-14_20_45': 'eLN x4/3',
	'2022_11_10-12_24_24': 'iLN x4/3',
	'2022_11_9-15_51_49': 'PN x4/3',
}

d_0p5_timetags = {
	'2022_11_10-17_17_4': 'ORN x1/2',
	'2022_11_10-17_17_35': 'eLN x1/2',
	'2022_11_10-17_17_41': 'iLN x1/2',
	'2022_11_10-17_17_50': 'PN x1/2',

	'2022_11_10-17_18_14': 'ORN x2',
	'2022_11_10-17_18_20': 'eLN x2',
	'2022_11_10-17_18_43': 'iLN x2',
	'2022_11_10-17_18_50': 'PN x2',
}

d_0p75_timetags = {
	'2022_11_10-17_24_27': 'ORN x1/4',
	'2022_11_10-17_24_34': 'eLN x1/4',
	'2022_11_11-9_42_0': 'iLN x1/4',
	'2022_11_11-10_31_7': 'PN x1/4',

	'2022_11_10-17_24_58': 'ORN x4',
	'2022_11_10-17_26_55': 'eLN x4',
	'2022_11_10-17_27_23': 'iLN x4',
	'2022_11_10-17_27_30': 'PN x4',
}

timetag_sets = [base_timetag, d_0p25_timetags, d_0p5_timetags, d_0p75_timetags]
timetag_set_paths_local = ['sensitivity_base',
					  'sensitivity_0p25/single_param', 
					  'sensitivity_0p5/single_param', 
					  'sensitivity_0p75/single_param'] 
timetag_set_paths_abs = [os.path.join(saved_sims_dir, d) for d in timetag_set_paths_local]


n_sets = len(timetag_sets)

for iis in range(n_sets):

	cur_timetag_set = timetag_sets[iis]
	cur_timetag_local = timetag_set_paths_local[iis]
	cur_timetag_path = timetag_set_paths_abs[iis]

	print('processing {}...'.format(cur_timetag_local))

	sim_dir_path = os.path.join(saved_sims_dir, cur_timetag_local)

	sim_dirs = os.listdir(sim_dir_path)

	run_timetags = list(cur_timetag_set.keys())
	run_names = list(cur_timetag_set.values())

	n_runs = len(run_timetags)

	d_sensitivity_res = {}
	for ir in range(n_runs):
		cur_timetag = run_timetags[ir]
		cur_name = run_names[ir]

		run_dir_local = [d for d in sim_dirs if cur_timetag in d][0]
		run_dir_abs = os.path.join(sim_dir_path, run_dir_local)

		run_dir_files = [os.path.join(run_dir_abs, f) for f in os.listdir(run_dir_abs)]
		run_df_AL_activity_f = [f for f in run_dir_files if 'df_AL_activity' in f][0]
		run_df_AL_activity = pd.read_csv(run_df_AL_activity_f)

		d_sensitivity_res[cur_name] = run_df_AL_activity

	res_file = os.path.join(sensitivity_analysis_dir, 'analysis/',
							cur_timetag_local.replace('/', '')+'_df_AL_activitys.p')
	pickle.dump(d_sensitivity_res, open(res_file, 'wb'))