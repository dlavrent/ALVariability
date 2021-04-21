import os
import sys
from pathlib import Path

file_path = os.path.abspath(__file__)

project_dir = os.path.join(file_path.split('ALVariability')[0], 'ALVariability')

sys.path.append(project_dir)




#hemi_dir = os.path.join(file_path.split('hemi3')[0], 'hemi3')
#p = Path(file_path)
#submissions_dir = os.path.join(hemi_dir, p.relative_to(hemi_dir).parts[0])
#sys.path.append(almodel_dir); sys.path.append(hemi_dir); sys.path.append(submissions_dir)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
from PIL import Image

from utils.plot_utils import match_ORN_fr_sim_to_input, plot_AL_psths, plot_neur_I_contribs,\
    plot_PN_ORN_fr_heatmaps, plot_PN_ORN_histograms, plot_glom_resmat, get_AL_psths,\
    plot_sim_spikes, plot_AL_activity_dur_pre_odors, plot_sim_ORN_PN_firing_rates, plot_orn_ln_pn_scaled_conmat
from utils.data_utils import get_AL_activity_dfs, get_orn_pn_frs_from_df_AL_activity, make_d_neur_inputs
from utils.simulation_class import Sim
from utils.LIF_funcs_cython import run_LIF_general_sim_wrap as run_LIF, spikes_from_APMask
from utils.data_utils import set_connect_from_scale_dic
from utils.model_params import params as default_params
from scipy.sparse import csr_matrix
from make_output_pdf import process_jdir

showplots=False
saveplots=True

#np.random.seed(124)

saveto_dir = os.path.dirname(os.path.abspath(__file__))

# default hemibrain data
hemibrain_dir = os.path.join(project_dir, 'connectomics/hemibrain_v1_2')
df_char_ids = pd.read_csv(os.path.join(hemibrain_dir, 'df_neur_ids.csv'), index_col=0)
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
    'df_char_ids': df_char_ids,
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


# plot results
# image saving settings
DPI = 300
fname_con = os.path.join(saveto_dir, 'sim_con.png')
fname_scaled_con = os.path.join(saveto_dir, 'sim_scaled_con.png')
fname_scaled_con_pdf = os.path.join(saveto_dir, 'sim_scaled_con.pdf')
fname_spikes = os.path.join(saveto_dir, 'spikes.png')
fname_frs = os.path.join(saveto_dir, 'firing_rates.png')
fname_pn_orn_fr_hists = os.path.join(saveto_dir, 'pn_orn_fr_hists.png')
fname_psths = os.path.join(saveto_dir, 'sim_PSTHs.png')

# plot connectivity
fb = sim.connect.copy(); fb[fb > 0] = 1; fb[fb < 0] = -1
plt.figure(figsize=(18,18))
sns.heatmap(fb, cmap='bwr', center=0, vmax=1.3, vmin=-1.3, cbar=True, 
            cbar_kws = {'label': 'sign of connection'})
if saveplots:
    plt.savefig(fname_con, bbox_inches='tight', dpi=DPI)
if showplots:
    plt.show()
    
# plot fancier connectivity 
fig = plt.figure(figsize=(16,16))
alblock = pd.DataFrame(np.abs(sim.connect), index=sim.df_neur_ids.bodyId, columns=sim.df_neur_ids.bodyId)
plot_orn_ln_pn_scaled_conmat(fig, sim.df_neur_ids, alblock)
if saveplots:
    plt.savefig(fname_scaled_con, bbox_inches='tight', dpi=DPI)
    #plt.savefig(fname_scaled_con_pdf, bbox_inches='tight')
if showplots:
    plt.show()

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
sim.df_char_ids.to_csv(os.path.join(saveto_dir, 'df_char_ids.csv'))

# get firing rate info
df_AL_activity, df_AL_activity_long = get_AL_activity_dfs(sim, Spikes)
# save info
df_AL_activity.to_csv(os.path.join(saveto_dir, 'df_AL_activity.csv'))
df_orn_frs, df_pn_frs, max_fr = get_orn_pn_frs_from_df_AL_activity(df_AL_activity, sim)
# get orn, pn firing rates
ex_orn_frs, ex_pn_frs = sim.get_ORN_PN_firing_rates(Spikes)

# plot spikes
plt.figure(figsize=(16,16))
plot_sim_spikes(sim, Spikes, False)
if saveplots:
    plt.savefig(fname_spikes, bbox_inches='tight', dpi=DPI)
if showplots:
    plt.show()

# plot AL neuron firing rates pre/dur odors
plt.figure(figsize=(11,4))
plt.subplot(121)
plot_AL_activity_dur_pre_odors(df_AL_activity_long)
# plot PN vs. ORN firing rates
plt.subplot(122)
plt.axis('equal')
plot_sim_ORN_PN_firing_rates(sim, df_AL_activity)#, show_pts=True)
plt.legend(title='odors', 
           bbox_to_anchor=(0, -0.2), 
           loc='upper left', borderaxespad=0)
plt.subplots_adjust(wspace=0.4)
if saveplots:
    plt.savefig(fname_frs, bbox_inches='tight', dpi=DPI)
if showplots:
    plt.show()
plt.close()

# plot heatmaps
plot_PN_ORN_fr_heatmaps(df_orn_frs, df_pn_frs, sim.df_char_ids, max_fr, saveto_dir=saveto_dir, showplot=showplots)

# plot hist
plt.figure(figsize=(10,4))
plot_PN_ORN_histograms(df_orn_frs, df_pn_frs, max_fr)
if saveplots:
    plt.savefig(fname_pn_orn_fr_hists, bbox_inches='tight', dpi=DPI)
if showplots:
    plt.show()
plt.close()

# plot psths
orn_psths, pn_psths, eln_psths, iln_psths = get_AL_psths(sim, Spikes)
plot_AL_psths(orn_psths, pn_psths, eln_psths, iln_psths, sim, 6)
if saveplots:
    plt.savefig(fname_psths, bbox_inches='tight', dpi=DPI)
if showplots:
    plt.show()
plt.close()

# plot correspondence of ORN spikes and odor input spikes
plt.figure(figsize=(11,5))
plt.subplot(122)
match_ORN_fr_sim_to_input(sim, Spikes)
plt.subplots_adjust(wspace=0.3)
if saveplots:
    plt.savefig(os.path.join(saveto_dir, 'match_orn_frs.png'), bbox_inches='tight', dpi=DPI)
if showplots:
    plt.show()
plt.close()


# plot inputs
d_neur_inputs = make_d_neur_inputs(sim, pd.DataFrame(sim.connect))

# ORN 
plt.figure(figsize=(12,8))
plot_neur_I_contribs(sim.ORNpos[0], d_neur_inputs, sim, I, Iin, V, Spikes)
if saveplots:
    plt.savefig(os.path.join(saveto_dir, 'Icontrib_ORN0.png'), bbox_inches='tight', dpi=DPI)
if showplots:
    plt.show()
plt.close()   

# iLN
plt.figure(figsize=(12,8))
plot_neur_I_contribs(sim.iLNpos[0], d_neur_inputs, sim, I, Iin, V, Spikes)
if saveplots:
    plt.savefig(os.path.join(saveto_dir, 'Icontrib_iLN0.png'), bbox_inches='tight', dpi=DPI)
if showplots:
    plt.show()
plt.close()   

# eLN
if sim.neLNs > 0:
    plt.figure(figsize=(12,8))
    plot_neur_I_contribs(sim.eLNpos[0], d_neur_inputs, sim, I, Iin, V, Spikes)
    if saveplots:
        plt.savefig(os.path.join(saveto_dir, 'Icontrib_eLN0.png'), bbox_inches='tight', dpi=DPI)
    if showplots:
        plt.show()
    plt.close()   

# PN
plt.figure(figsize=(12,8))
plot_neur_I_contribs(sim.PNpos[60], d_neur_inputs, sim, I, Iin, V, Spikes)
if saveplots:
    plt.savefig(os.path.join(saveto_dir, 'Icontrib_PN60.png'), bbox_inches='tight', dpi=DPI)
if showplots:
    plt.show()
plt.close()   

# plot 
plot_glom_resmat('D', I, sim)
if saveplots:
    plt.savefig(os.path.join(saveto_dir, 'glom_D_resmat.png'), bbox_inches='tight', dpi=DPI)
if showplots:
    plt.show()
plt.close()   

print('saving pdf of images:')

# set pictures to save to PDF
imgs_for_pdf = [fname_spikes, fname_frs, fname_pn_orn_fr_hists,
                os.path.join(saveto_dir, 'orn_fr_heatmap.png'),
                os.path.join(saveto_dir, 'pn_fr_heatmap.png'),
                fname_psths]

# open files, convert to RGB
images = []
for f in imgs_for_pdf:    
    im = Image.open(f)
    if im.mode == "RGBA":
        im = im.convert("RGB")
    images.append(im)

# save PDF
pdfname = 'sim_plots_{}.pdf'.format(PARAMS['run_tag'])
images[0].save(os.path.join(saveto_dir, pdfname),
               save_all = True, quality = 70, append_images = images[1:])


# save BETTER PDF
all_pdf_fpath = os.path.join(submissions_dir, 'all_pdfs_resample6')
if not os.path.exists(all_pdf_fpath):
    os.makedirs(all_pdf_fpath)
print('making BETTER pdf...')
print(f'saving here and to {all_pdf_fpath}')
process_jdir(saveto_dir, 'output_figs', all_pdf_fpath)

if PARAMS['erase_sim_output']:
	# clear out the big stuff in sim.
    sim.Iin_rates = 0
    sim.Iin_PSCs = 0
    sim.chou_df = []
    sim.odor_stimulus = 0
    sim.all_LNs = []
    sim.df_door_response_matrix = []
    sim_outputs['sim'] = sim
    pickle.dump(sim_outputs, open(os.path.join(saveto_dir, 'sim_output.p'), 'wb'))