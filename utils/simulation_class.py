# -*- coding: utf-8 -*-
"""
This file defines the Sim class for simulating the Drosophila AL
December 2019
"""

import numpy as np
import sys
import pandas as pd
sys.path.append('..')
from utils.odor_utils import load_door_data
from utils.model_params import params as default_params
from utils.LIF_funcs_cython import make_PSC
from utils.data_utils import set_params


class Sim(object):
    '''
    Class used for storing information and 
    executing actions for AL simulation
    '''
    
    def __init__(self, params, df_neur_ids, al_block,
                 model_name='',
                 end_time = 1, sim_dt = 1e-4,
                 home_dir = ''):
        '''
        Initialize a simulation with a 
            model name (model_name),
            glomerulus + neuron physiological parameters (params),
            connectivity strength (connect_params),
            end time in seconds (end_time),
            timestep in seconds (sim_dt),
            path to home directory (where ALModel/ is)
        '''
        
        # load params -- required argument
        self.params = params
            
        # set empty connectivity parameters, 
        # use self.set_connect_params() to fill in later
        self.connect_params = {}

        # model name
        self.model_name = model_name
        
        # set time information
        self.end_time = end_time
        self.dt = sim_dt
        self.Nsteps = int(self.end_time / self.dt)
        self.time = np.arange(0, self.end_time, self.dt)
        
        # load matrices for LN wiring, odor responses   
        self.df_door_mappings, \
            self.df_door_odor, \
            self.df_door_response_matrix, \
            self.df_odor_response_matrix = load_door_data(home_dir)
        
        self.df_neur_ids = df_neur_ids
        self.al_block_df = al_block
        
        # define glomeruli
        orn_gloms = df_neur_ids[df_neur_ids['altype'] == 'ORN']['glom'].value_counts().index.values
        pn_gloms = df_neur_ids[df_neur_ids['altype'] == 'uPN']['glom'].value_counts().index.values
        self.glom_names = np.intersect1d(orn_gloms, pn_gloms)
        self.ngloms = len(self.glom_names)
        
        # set number of neuron types
        self._set_params(params)
        
        # use the receptors of the defined glomeruli to parse DoOR data
        self.glom_df = (self.df_door_mappings
                        [(self.df_door_mappings.code.isin(self.glom_names)) &
                         (self.df_door_mappings.index.isin(self.df_odor_response_matrix.columns))]
                        .reset_index()
                        .set_index('code')
                        .rename_axis('model_glom')
                        .reindex(self.glom_names))
        # DoOR receptor names of the defined glomeruli
        self.glom_receptors = self.glom_df.receptor.fillna('?').values
        # 'glomerulus (receptor)' labels
        self.glom_labels = self.glom_df['receptor'].reset_index().fillna('?').apply(lambda x: '{} ({})'.format(*x[::-1][::-1]), axis = 1).values
        
        # initialize odor stimulus array
        self.odor_stimulus = np.zeros((self.nORNs, self.Nsteps))
        self.odor_list = []
        self.all_odor_door_responses = []
        self.Iin_rates = np.zeros((self.nORNs, self.Nsteps))
        self.Iin_PSCs = np.copy(self.odor_stimulus)

    def __str__(self):
        '''
        Print some model information
        '''
        return "Model: {} \n".format(self.model_name) + \
            'Simulation time: {} s, dt: {} s'.format(self.end_time, self.dt)
                
    def __repr__(self):
        return str(self)
    
        
    def set_eLNs(self, eln_set):
        '''
        Given an array of positions of the eLN set in the LNs
        (i.e. values from 0 to self.nLNs),
        stores the positions of iLNs/eLNs
        '''
        iln_set =  np.array([x for x in np.arange(self.nLNs) \
                             if x not in eln_set])
        self.iLNpos = self.LNpos[iln_set]
        self.eLNpos = self.LNpos[eln_set]
        self.niLNs = len(self.iLNpos)
        self.neLNs = len(self.eLNpos)
        self.neur_types[self.iLNpos] = 'iLN'
        self.neur_types[self.eLNpos] = 'eLN'
        self.df_neur_ids.loc[self.eLNpos, 'polarity'] = +1
        
    
    def _set_params(self, params):
        '''
        Simply calls set_params(self, params) from data_utils
        '''
        set_params(self, params)
        self.iLNpos = self.LNpos
        self.eLNpos = np.array([])
        self.niLNs = len(self.iLNpos)
        self.neLNs = len(self.eLNpos)
        self.neur_types = np.array(['ORN']*self.nAL)
        self.neur_types[self.PNpos] = 'PN'
        self.neur_types[self.LNpos] = 'iLN'
        self.df_neur_ids.loc[self.LNpos, 'polarity'] = -1
        
        
    def set_connect(self, connect, check_signs=True):
        '''
        Given a connectivity matrix of size nAL * nAL,
        first checks that dimensions are correct,
        then checks that signs are correct 
        (ORNs/eLNs/PNs excitatory, iLNs inhibitory),
        then stores it as self.connect
        '''
        if np.prod(connect.shape) != self.nAL**2:
            print('shape of connect is {}x{}, but should be {}x{}!'\
                  .format(*connect.shape, self.nAL, self.nAL))
            return -1
        
        # assign polarity (iLNs, GABAergic PNs should be negative)
        if check_signs:            
            neurons_pos_polarity = self.df_neur_ids[self.df_neur_ids.polarity == +1].index.values
            neurons_neg_polarity = self.df_neur_ids[self.df_neur_ids.polarity == -1].index.values
            connect[neurons_pos_polarity, :] = np.abs(connect[neurons_pos_polarity, :] )  
            connect[neurons_neg_polarity, :] = -np.abs(connect[neurons_neg_polarity, :] )   
        
        self.connect = connect
        
    def get_odor_names(self):
        '''
        Returns list of odors (each added through add_odor())
        '''
        return [r[0] for r in self.odor_list]
    
    def get_odor_positions(self):
        '''
        For each odor in odor_list, return the time indices
        '''
        ts = []
        for row in self.odor_list:
            odor_name, odor_start, odor_end = row
            odor_start_pos = np.where(self.time >= odor_start)[0][0]
            odor_end_pos = np.where(self.time > odor_end)[0][0]
            ts.append([odor_start_pos, odor_end_pos])
        return ts
            
    def get_odor_firing_rates(self, Spikes, odor_index=0):
        '''
        Given a (num_neurons x num_timepoints) Spike array,
        and an odor of interest (odor_index),
        finds the relevant time points in the Spike array 
        to give firing rate under that odor for all neurons
        '''
        odor_info = self.odor_list[odor_index]
        odor_name, odor_start, odor_end = odor_info
        #print(odor_info)
        odor_start_pos = np.where(self.time >= odor_start)[0][0]
        odor_end_pos = np.where(self.time > odor_end)[0][0]
        odor_dur = odor_end - odor_start
        return Spikes[:, odor_start_pos:odor_end_pos].sum(1) / odor_dur
    
    def get_pre_odor_firing_rates(self, Spikes, odor_index=0):
        '''
        Given a (num_neurons x num_timepoints) Spike array,
        and an odor of interest (odor_index),
        finds the relevant time points in the Spike array 
        to give firing rate BEFORE that odor for all neurons
        '''
        odor_name, odor_start, odor_end = self.odor_list[odor_index]
        # get the start of the PAUSE before the odor
        if odor_index == 0:
            # if first odor, start at t=0
            pause_start = 0
        else:
            # if not, get the end time of previous odor
            pause_start = self.odor_list[odor_index-1][2]
        pause_start_pos = np.where(self.time >= pause_start)[0][0]
        odor_start_pos = np.where(self.time > odor_start)[0][0]
        odor_pause = odor_start - pause_start
        return Spikes[:, pause_start_pos:odor_start_pos].sum(1) / odor_pause
    
    def get_ORN_PN_firing_rates(self, Spikes, odor_index=0):
        '''
        Given a (num_neurons x num_timepoints) Spike array,
        and an odor of interest (odor_index),
        finds the relevant time points in the Spike array 
        to give firing rate under that odor for ORNs and PNs only
        '''
        odor_firing_rates = self.get_odor_firing_rates(Spikes, odor_index)
        return odor_firing_rates[self.ORNpos], odor_firing_rates[self.PNpos]
    
    def get_ORN_PN_firing_rates_mean_class(self, Spikes, odor_index=0):
        '''
        Given a (num_neurons x num_timepoints) Spike array,
        and an odor of interest (odor_index),
        finds the relevant time points in the Spike array 
        to give firing rate under that odor for ORNs and PNs only
        '''
        rates = self.get_odor_firing_rates(Spikes, odor_index)
        return np.mean(rates[self.ORNpos]), \
            np.mean(rates[self.LNpos]), \
            np.mean(rates[self.PNpos])
            
    def add_odor(self, odor_name, odor_start, odor_end, 
                 tc=1e10, fadapt=0.5, 
                 manual_odor_glom_responses=[],
                 imputed_glom_responses=[]):
        '''
        Given common name of odor (odor)name),
        looks it up in the DoOR response table (already stored in this class),
        finds the glomerular responses, 
        adds it to odor_stimulus
        '''
        # do some sanity checks first
        if odor_start < 0:
            print('odor start ({:.2f} s) is negative!')
            return -1
        if odor_start > self.end_time:
            print('odor start ({:.2f} s) is after simulation time ({:.2f} s)!'.format(odor_start, self.end_time))
            return -1
        if odor_end > self.end_time:
            print('odor end ({:.2f} s) is after simulation time ({:.2f} s)!'.format(odor_end, self.end_time))
            return -1
        # retrieve glomerular responses for the odor (between 0 and 1) from DOOR
        if len(manual_odor_glom_responses) == 0:
            or_response = (self.df_odor_response_matrix
                           .T[odor_name]
                           .rename_axis('receptor')
                           .reset_index())
            odor_glom_responses = (self.glom_df
                                   .reset_index()
                                   .merge(or_response, how='left')
                                   .set_index('model_glom')
                                   [odor_name]
                                   .fillna(0))
            # use imputed table instead if available
            if len(imputed_glom_responses) > 0:
                odor_glom_responses = (imputed_glom_responses
                           .loc[self.glom_names, odor_name]
                           .rename_axis('model_glom'))
        else:
            # add a custom odor not necessarily in DoOR
            odor_glom_responses = manual_odor_glom_responses
        
        
        odor_orn_responses = (self.df_neur_ids
              [self.df_neur_ids.altype=='ORN']
              .merge(pd.DataFrame(odor_glom_responses), 
                     left_on='glom', 
                     right_on=odor_glom_responses.index.name, 
                     how='left')
              [odor_name].values)
        #self.all_odor_door_responses.append(odor_glom_responses)        
        odor_times = (np.arange(odor_start/self.dt, 
                            odor_end/self.dt)).astype(int)
        for i in range(self.nORNs):
            max_val = odor_orn_responses[i]
            t_arr = (odor_times - odor_times[0])*self.dt
            # only add a decay if there's a positive odor response!
            if max_val > 0:
                exp_decay = fadapt*max_val + (1-fadapt)*max_val*np.exp(-t_arr/tc)
            else:
                exp_decay = max_val
            self.odor_stimulus[i, odor_times] = exp_decay
            
        # update input current
        self.set_Iin_rates()
        # add to internal odor list
        self.odor_list.append([odor_name, odor_start, odor_end])
        
        return odor_glom_responses, odor_orn_responses, odor_times
    
    def draw_Iin_PSCs(self):
        '''
        Once rates are established with self.set_Iin_rates(),
        actually draw instances of spikes, for use in make_Iin()
        '''
        Iin_rates_per_dt = self.Iin_rates * self.dt
        ran_vals = np.random.uniform(0, 1, Iin_rates_per_dt.shape)
        return (ran_vals < Iin_rates_per_dt).astype(int)
        
    
    def set_Iin_rates(self):
        '''
        Given baseline ORN rate and the odor stimulus,
        sets the input current firing rates
        Should be run at least after the last odor.
        '''
        # look up the baseline ORN rate, 
        # and then contribution from odors
        # (look up the DoOR response 0-1 in self.odor_stimulus, 
        # and then just multiply by the set maximum odor rate)
        self.Iin_rates[self.ORNpos, :] = self.params['spon_fr_ORN']  + \
                self.params['odor_rate_max'] * self.odor_stimulus
        # in case a baseline rate - odor-evoked rate is negative, set to 0
        self.Iin_rates[self.Iin_rates < 0] = 0
       
    
    def make_Iin(self, is_input_PSC=False):
        '''
        Generates input current array for the ORNs
        based on the rates set by the odor stimulus
        '''
        
        # draw template PSCs for all neurons
        all_PSCs = [make_PSC(self, i) for i in range(self.nAL)]
        # find maximum buffer time to add at the end of simulation
        # (for, say, a PSC that starts at the end of the time array)
        max_tBuff = max([len(PSC) for PSC in all_PSCs])
        
        # draw input PSCs onto ORNs
        # if is_input_PSC == True, pull from params file for duration
        # if False, then use the settings for the ORN PSC duration
        orn_input_PSCs = [make_PSC(self, i, is_input_PSC) for i in range(self.nORNs)]
        
        # initialize input current array
        Iin = np.zeros((self.nAL, self.Nsteps+max_tBuff))
        # fill it in with presence of PSCs (this array is binary 0/1)
        Iin_PSCs = self.draw_Iin_PSCs()
        
        
        for i in range(self.nORNs):
            #print(i)
            # get neuron-specific PSC template
            PSCi = orn_input_PSCs[i]
            tBuffi = len(PSCi)
            if min(Iin_PSCs[i, :]) == 0:
                pscFallTimes = np.where(Iin_PSCs[i, :] == 1)[0]
            else:
                pscFallTimes = Iin_PSCs[i, :]
            for j in range(len(pscFallTimes)):
                PSCrange = pscFallTimes[j] + np.arange(tBuffi)
                Iin[self.ORNpos[i], PSCrange] += PSCi*self.params['PSCweight_ORN']

        # finally, add PSC buffer time to the time array
        self.time = np.arange(0, self.Nsteps+max_tBuff)*self.dt
        return Iin
  
    
    
'''   
    
import os
import sys

project_dir = '../'

sys.path.append(project_dir)

#from utils.LIF_funcs import run_LIF, spikes_from_APMask

from utils.LIF_funcs_cython import run_LIF_general_sim_wrap as run_LIF, spikes_from_APMask


from utils.data_utils import set_connect_from_scale_dic



df_neur_ids = pd.read_csv(os.path.join(project_dir, 'connectomics/hemibrain_v1_2/df_neur_ids.csv'), index_col=0)
al_block = pd.read_csv(os.path.join(project_dir, 'connectomics/hemibrain_v1_2/al_block.csv'), index_col=0)

odor_panel = ['4-methylcyclohexanol', '3-octanol']
odor_dur = 0.02
odor_pause = 0.01
end_padding = 0.001
params = default_params
elnpos = np.arange(31)


# ### Set up odor deliveries
trial_dur = odor_dur + odor_pause
end_time = (len(odor_panel)*trial_dur) + end_padding


# ### Set up simulation

# initialize simulation
sim = Sim(params = params,
          df_neur_ids=df_neur_ids,
          al_block=al_block,
          model_name = 'hemi_all',
          home_dir = project_dir,
          end_time = end_time
      )


# set up timings
t_odor_bounds = np.arange(0, end_time+0.01, trial_dur)
start_times = t_odor_bounds[:-1] + odor_pause
end_times = t_odor_bounds[1:]

# add odors according to their times
for i in range(len(odor_panel)):
    sim.add_odor(odor_name = odor_panel[i],
                 odor_start = start_times[i], 
                 odor_end = end_times[i])

# set Iin
Iin = sim.make_Iin(is_input_PSC=True)

# ### Set synapse strengths    
sim.set_eLNs(elnpos)
# set scalars on class to class strengths
custom_scale_dic = {}
connect = set_connect_from_scale_dic(custom_scale_dic, sim)
sim.set_connect(connect.values)

print('running LIF')
# ### Run simulation
V, I, APMask = run_LIF(sim, Iin)

# count spikes
Spikes = spikes_from_APMask(APMask)



df_AL_activity, df_AL_activity_long = get_AL_activity_dfs(sim, Spikes)




      
if __name__ == 'main':
    
    import matplotlib.pyplot as plt
    import seaborn as sns
        
        
    home_dir = 'C:/Users/dB/deBivort/projects/ALModel/'
            
    #### SET GLOMERULI
            
    
    hemi_params = default_params.copy()
            
    
    #### SET ODORS
    
    # odors, sequentially
    odor_panel = ['4-methylcyclohexanol', 
                  #'3-octanol',
                  #'ethyl lactate',
                  #'2-heptanone',
                  #'geranyl acetate',
    ]
    
    # duration of odor stimulus (seconds)
    odor_dur = 0.2
    
    # break between odor stimuli (seconds)
    odor_pause = 0.2
    
    trial_dur = odor_dur + odor_pause
    end_time = (len(odor_panel)*trial_dur)*1.01
    
    # initialize simulation
    sim = Sim(params = hemi_params,
          model_name = 'hemibrain',
          home_dir = home_dir,
          end_time = end_time
          )
        
    
    
    t_odor_bounds = np.arange(0, end_time+0.01, trial_dur)
    start_times = t_odor_bounds[:-1] + odor_pause
    end_times = t_odor_bounds[1:]
    
    # add odors according to their times
    for i in range(len(odor_panel)):
        sim.add_odor(odor_name = odor_panel[i],
                     odor_start = start_times[i], 
                     odor_end = end_times[i])
        
    #foo
    Iin = sim.make_Iin(is_input_PSC=True)

'''