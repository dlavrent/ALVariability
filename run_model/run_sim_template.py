import os
import sys
import numpy as np
import pandas as pd
import pickle
import time

file_path = os.path.abspath(__file__)
project_dir = os.path.join(file_path.split('ALVariability')[0], 'ALVariability')
sys.path.append(project_dir)

from utils.data_utils import get_AL_activity_dfs
from utils.simulation_class import Sim
from utils.LIF_funcs_cython import run_LIF_general_sim_wrap as run_LIF, spikes_from_APMask
from utils.data_utils import set_connect_from_scale_dic
from utils.model_params import params as default_params
from scipy.sparse import csr_matrix
from utils.plot_model_outputs import process_jdir


saveto_dir = os.path.dirname(os.path.abspath(__file__))

# default hemibrain data
hemibrain_dir = os.path.join(project_dir, 'connectomics/hemibrain_v1_2')
df_neur_ids = pd.read_csv(os.path.join(hemibrain_dir, 'df_neur_ids.csv'), index_col=0)
al_block = pd.read_csv(os.path.join(hemibrain_dir, 'AL_block.csv'), index_col=0)
   
PARAMS = {
	'run_tag': 'hemi_all',
    'odor_panel': ['3-octanol', '4-methylcyclohexanol'],
    'odor_dur': 0.01,
    'odor_pause': 0.005,
    'end_padding': 0.001,
    'hemi_params': default_params,
    'elnpos': np.array([]),
    'custom_scale_dic': {},
    'ln_add_current': 0,
    'decay_tc': 1e8,
    'decay_fadapt': 1,
    'erase_sim_output': 0,
    'imputed_glom_odor_table': [],
    'df_neur_ids': df_neur_ids,
    'al_block': al_block
    }


# load settings from seed
sim_params_seed_d = pickle.load( open( 'sim_params_seed.p', "rb" ) )
for k in sim_params_seed_d.keys():
    PARAMS[k] = sim_params_seed_d[k]


t0 = time.time()


############## SIMULATION CODE

# ### Set up odor deliveries
n_odors = len(PARAMS['odor_panel'])
trial_dur = PARAMS['odor_dur'] + PARAMS['odor_pause']
end_time = (n_odors*trial_dur) + PARAMS['end_padding']


# ### Set up simulation

# initialize simulation
sim = Sim(params = PARAMS['hemi_params'],
      df_neur_ids=PARAMS['df_neur_ids'], 
      al_block=PARAMS['al_block'],
      model_name = PARAMS['run_tag'],
      home_dir = project_dir,
      end_time = end_time,
      )

# set up timings
t_odor_bounds = np.arange(0, end_time+0.01, trial_dur)
start_times = t_odor_bounds[:-1] + PARAMS['odor_pause']
end_times = t_odor_bounds[1:]

# add odors according to their times
for i in range(n_odors):
    sim.add_odor(odor_name = PARAMS['odor_panel'][i],
                 odor_start = start_times[i], 
                 odor_end = end_times[i],
                 tc = PARAMS['decay_tc'],
                 fadapt = PARAMS['decay_fadapt'],
                 imputed_glom_responses=PARAMS['imputed_glom_odor_table'])
        
# set Iin
Iin = sim.make_Iin(is_input_PSC=True)
# add Iin current to LNs
Iin[sim.LNpos, :] = PARAMS['ln_add_current']

# ### Set synapse strengths    
sim.set_eLNs(PARAMS['elnpos'])
# set scalars on class to class strengths
connect = set_connect_from_scale_dic(PARAMS['custom_scale_dic'], sim)
sim.set_connect(connect.values)   
    
# ### Run simulation
print('running LIF...')
V, I, APMask = run_LIF(sim, Iin)
print('done running LIF')

# count spikes
Spikes = spikes_from_APMask(APMask)

tf = time.time()
print('elapsed', tf-t0)

# save simulation results
sim_outputs = {}
sim_outputs['sim'] = sim
sim_outputs['V'] = V
sim_outputs['I'] = csr_matrix(I)
sim_outputs['Iin'] = csr_matrix(Iin)
sim_outputs['Spikes'] = csr_matrix(Spikes)

pickle.dump(sim_outputs, open(os.path.join(saveto_dir, 'sim_output.p'), 'wb'))
pickle.dump(csr_matrix(sim.connect), open(os.path.join(saveto_dir, 'sim_connect.p'), 'wb'))
pickle.dump(sim.neur_names, open(os.path.join(saveto_dir, 'sim_neur_names.p'), 'wb'))
pickle.dump(csr_matrix(Spikes), open(os.path.join(saveto_dir, 'Spikes_csr.p'), 'wb'))
sim.df_neur_ids.to_csv(os.path.join(saveto_dir, 'df_neur_ids.csv'))

# get firing rate info
df_AL_activity, df_AL_activity_long = get_AL_activity_dfs(sim, Spikes)
# save info
df_AL_activity.to_csv(os.path.join(saveto_dir, 'df_AL_activity.csv'))


# save model output PDF
run_model_dir = os.path.join(file_path.split('run_model')[0], 'run_model')
all_pdf_fpath = os.path.join(run_model_dir, 'all_pdfs_sensitivity_sweep')
if not os.path.exists(all_pdf_fpath):
    os.makedirs(all_pdf_fpath)
print('making pdf...')
print(f'saving here and to {all_pdf_fpath}')

process_jdir(saveto_dir, 'output_figs', all_pdf_fpath)

if PARAMS['erase_sim_output']:
	# clear out the big stuff in sim.
    sim.Iin_rates = 0
    sim.Iin_PSCs = 0
    sim.odor_stimulus = 0
    sim.all_LNs = []
    sim.df_door_response_matrix = []
    sim_outputs['sim'] = sim
    pickle.dump(sim_outputs, open(os.path.join(saveto_dir, 'sim_output.p'), 'wb'))