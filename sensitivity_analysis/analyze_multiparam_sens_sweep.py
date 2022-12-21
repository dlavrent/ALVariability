import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import re
import os
import sys

file_path = os.path.abspath(__file__)
project_dir = os.path.join(file_path.split('ALModel_dev')[0], 'ALModel_dev')

sensitivity_analysis_dir = os.path.join(project_dir, 'sensitivity_analysis')
saved_sims_dir = os.path.join(sensitivity_analysis_dir, 'save_sims_sensitivity_sweep/multi_0p5')

sim_dirs = [x for x in os.listdir(saved_sims_dir) if '__0v12_' in x]


n_sim_dirs = len(sim_dirs)

n_success = 0
d_res = {}
for iis in range(n_sim_dirs):
	cur_sim_dir = sim_dirs[iis]
	cur_sim_dir_abs = os.path.join(saved_sims_dir, cur_sim_dir)
	sim_dir_files = os.listdir(cur_sim_dir_abs)

	sim_dir_files = [os.path.join(cur_sim_dir_abs, f) for f in os.listdir(cur_sim_dir_abs)]
	sim_df_AL_activity_f = [f for f in sim_dir_files if 'df_AL_activity' in f]
	print(sim_df_AL_activity_f)
	if len(sim_df_AL_activity_f) == 1:
		sim_df_AL_activity_f = sim_df_AL_activity_f[0]
		sim_df_AL_activity = pd.read_csv(sim_df_AL_activity_f)

		time_tag = cur_sim_dir.split('__0v12')[0]
		re_matches = re.findall('_all([0-9]*[.]?[0-9]+)_ocol([0-9]*[.]?[0-9]+)_ecol([0-9]*[.]?[0-9]+)_icol([0-9]*[.]?[0-9]+)_pcol([0-9]*[.]?[0-9]+)', cur_sim_dir)[0]
		all_col, orn_col, eln_col, iln_col, pn_col = [float(x) for x in re_matches]

		d_res['0p5_'+time_tag] = {'all': all_col, 'orn': orn_col, 'eln': eln_col, 'iln': iln_col, 'pn': pn_col,
							  'df_AL_activity': sim_df_AL_activity}

		print('adding {}'.format(cur_sim_dir))
		n_success += 1

print('{} successful out of {}'.format(n_success, n_sim_dirs))

res_file = os.path.join(sensitivity_analysis_dir, 'analysis/',
						'multiparam_0p5_df_AL_activitys.p')
pickle.dump(d_res, open(res_file, 'wb'))