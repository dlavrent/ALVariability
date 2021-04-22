"""
Defines utility functions for setting up simulation class,
namely names and positions of neuron types
"""

import numpy as np
import pandas as pd

def get_neuron_ns_pos(df_neur_ids):
    '''
    Given glomerulus names, and number of neuron types per glomerulus,
    returns numbers of neuron types, 
        their indexing positions assuming order is ORNs, LNs, PNs,
        and names of ORNs, LNs, PNs
    '''
    
    tabORN = df_neur_ids[df_neur_ids.altype == 'ORN']
    tabLN = df_neur_ids[df_neur_ids.altype == 'LN']
    tabuPN = df_neur_ids[df_neur_ids.altype == 'uPN']
    tabmPN = df_neur_ids[df_neur_ids.altype == 'mPN']
    
    nORNs = len(tabORN); nLNs = len(tabLN); nuPNs = len(tabuPN); nmPNs = len(tabmPN)
    nAL = nORNs + nLNs + nuPNs + nmPNs
    
    # get indexing positions
    ORNpos = tabORN.index.values #np.where(df_neur_ids.altype == 'ORN')[0] #np.arange(nORNs)
    LNpos = tabLN.index.values #np.where(df_neur_ids.altype == 'LN')[0] #nORNs + np.arange(nLNs)
    uPNpos = tabuPN.index.values #np.where(df_neur_ids.altype == 'PN')[0] #nORNs + nLNs + np.arange(nPNs)
    mPNpos = tabmPN.index.values #np.where(df_neur_ids.altype == 'PN')[0] #nORNs + nLNs + np.arange(nPNs)
        
    # set neuron names
    ORN_cumcnt = tabORN.groupby('glom').cumcount().values.astype(str)
    ORN_names = 'ORN_' + tabORN['glom'].astype(str) + '_' + ORN_cumcnt
    
    LN_names = [f'LN_{i}' for i in range(nLNs)]
    
    uPN_cumcnt = tabuPN.groupby('glom').cumcount().values.astype(str)
    uPN_names = 'uPN_' + tabuPN['glom'].astype(str) + '_' + uPN_cumcnt
    
    mPN_names = [f'mPN_{i}' for i in range(nmPNs)]
   
    # combine all names together
    neur_names = np.concatenate((ORN_names, LN_names, uPN_names, mPN_names))
    
    return nAL, nORNs, nLNs, nuPNs, nmPNs, \
        ORNpos, LNpos, uPNpos, mPNpos, \
        neur_names


def set_params(self, params):
        '''
        Given number of glomeruli specified in params, 
        and parameters resolved by ORN, LN, PN,
        updates self (a Sim instance) with vectors of parameters,
        with appropriate values for each neuron class
        '''
        self.nAL, \
            self.nORNs, self.nLNs, self.nuPNs, self.nmPNs, \
            self.ORNpos, self.LNpos, self.uPNpos, self.mPNpos, \
            self.neur_names = get_neuron_ns_pos(self.df_neur_ids)
        self.PNpos = np.concatenate((self.uPNpos, self.mPNpos))
        self.iLNpos = self.LNpos
        self.eLNpos = np.array([])
            
        self.neur_names = pd.Series(self.neur_names)
        
        self.params = params
        
        self.params['V0s'] = np.zeros(self.nAL)
        self.params['V0s'][self.ORNpos] = params['V0_ORN']
        self.params['V0s'][self.LNpos] = params['V0_LN']
        self.params['V0s'][self.PNpos] = params['V0_PN']
        
        self.params['Vthrs'] = np.zeros(self.nAL)
        self.params['Vthrs'][self.ORNpos] = params['Vthr_ORN']
        self.params['Vthrs'][self.LNpos] = params['Vthr_LN']
        self.params['Vthrs'][self.PNpos] = params['Vthr_PN']
        
        self.params['Vmaxs'] = np.zeros(self.nAL)
        self.params['Vmaxs'][self.ORNpos] = params['Vmax_ORN']
        self.params['Vmaxs'][self.LNpos] = params['Vmax_LN']
        self.params['Vmaxs'][self.PNpos] = params['Vmax_PN']
        
        self.params['Vmins'] = np.zeros(self.nAL)
        self.params['Vmins'][self.ORNpos] = params['Vmin_ORN']
        self.params['Vmins'][self.LNpos] = params['Vmin_LN']
        self.params['Vmins'][self.PNpos] = params['Vmin_PN']
        
        self.params['Cmems'] = np.zeros(self.nAL)
        self.params['Cmems'][self.ORNpos] = params['Cmem_ORN']
        self.params['Cmems'][self.LNpos] = params['Cmem_LN']
        self.params['Cmems'][self.PNpos] = params['Cmem_PN']
        
        self.params['Rmems'] = np.zeros(self.nAL)
        self.params['Rmems'][self.ORNpos] = params['Rmem_ORN']
        self.params['Rmems'][self.LNpos] = params['Rmem_LN']
        self.params['Rmems'][self.PNpos] = params['Rmem_PN']
        
        self.params['APdurs'] = np.zeros(self.nAL)
        self.params['APdurs'][self.ORNpos] = params['APdur_ORN']
        self.params['APdurs'][self.LNpos] = params['APdur_LN']
        self.params['APdurs'][self.PNpos] = params['APdur_PN']
        
        self.params['PSCmags'] = np.zeros(self.nAL)
        self.params['PSCmags'][self.ORNpos] = params['PSCmag_ORN']
        self.params['PSCmags'][self.LNpos] = params['PSCmag_LN']
        self.params['PSCmags'][self.PNpos] = params['PSCmag_PN']
        
        self.params['PSCrisedurs'] = np.zeros(self.nAL)
        self.params['PSCrisedurs'][self.ORNpos] = params['PSCrisedur_ORN']
        self.params['PSCrisedurs'][self.LNpos] = params['PSCrisedur_LN']
        self.params['PSCrisedurs'][self.PNpos] = params['PSCrisedur_PN']
        
        self.params['PSCfalldurs'] = np.zeros(self.nAL)
        self.params['PSCfalldurs'][self.ORNpos] = params['PSCfalldur_ORN']
        self.params['PSCfalldurs'][self.LNpos] = params['PSCfalldur_LN']
        self.params['PSCfalldurs'][self.PNpos] = params['PSCfalldur_PN']


def get_AL_activity_dfs(sim, Spikes):
    df_AL_activity = pd.DataFrame()
    # get names
    df_AL_activity['neur_name'] = sim.neur_names 
    df_AL_activity['neur_type'] = ''
    df_AL_activity['neur_type'].iloc[sim.ORNpos] = 'ORN'
    if len(sim.eLNpos) > 0:
        df_AL_activity['neur_type'].iloc[sim.iLNpos] = 'iLN'
        df_AL_activity['neur_type'].iloc[sim.eLNpos] = 'eLN'
    else:
        df_AL_activity['neur_type'].iloc[sim.LNpos] = 'LN'
    df_AL_activity['neur_type'].iloc[sim.uPNpos] = 'uPN'
    df_AL_activity['neur_type'].iloc[sim.mPNpos] = 'mPN'
    
    cur_t = 0
    for i in range(len(sim.odor_list)):    
        odor_name, odor_start, odor_end = sim.odor_list[i]
        
        odor_pause = odor_start - cur_t
        odor_dur = odor_end - odor_start
        pre_odor_t_indices = np.where((cur_t <= sim.time) & (sim.time < odor_start))[0]
        dur_odor_t_indices = np.where((odor_start <= sim.time) & (sim.time < odor_end))[0]
        df_AL_activity['fr_pre_odor{}'.format(i)] = np.sum(Spikes[:, pre_odor_t_indices], 1) / odor_pause
        df_AL_activity['fr_dur_odor{}'.format(i)] = np.sum(Spikes[:, dur_odor_t_indices], 1) / odor_dur
        cur_t = odor_end

    df_AL_activity_long = (df_AL_activity
       .melt(id_vars=['neur_name', 'neur_type'],
             var_name='timeframe', 
             value_name='fr')
    )
    df_AL_activity_long['dur_odor'] = df_AL_activity_long['timeframe'].str.contains('dur')
    df_AL_activity_long['odor_num'] = [int(n[-1]) for n in df_AL_activity_long['timeframe']]
    
    # add glom info
    df_AL_activity_long['glom'] = [ r.split('_')[1] if not (r.startswith('LN') or r.startswith('mPN')) else np.nan for r in df_AL_activity_long.neur_name]
    df_AL_activity['glom'] = [ r.split('_')[1] if not (r.startswith('LN') or r.startswith('mPN'))else np.nan for r in df_AL_activity.neur_name]
    
    return df_AL_activity, df_AL_activity_long


def get_orn_pn_frs_from_df_AL_activity(df_AL_activity, sim, sub_pre=False, olf_only=False):
    odor_names = [r[0] for r in sim.odor_list]
    #orn_gloms = df_neur_ids[df_neur_ids.altype=='ORN']['glom']
    #pn_gloms = df_neur_ids[df_neur_ids.altype=='PN']['glom']
    
    
    df_orn_activity = df_AL_activity.loc[df_AL_activity.neur_type == 'ORN'].set_index('neur_name')
    df_orn_frs = df_orn_activity.loc[:, df_orn_activity.columns.str.contains('dur')]
    df_orn_frs.columns = odor_names
    

    df_upn_activity = df_AL_activity.loc[df_AL_activity.neur_type == 'uPN'].set_index('neur_name')
    df_upn_frs = df_upn_activity.loc[:, df_upn_activity.columns.str.contains('dur')]
    df_upn_frs.columns = odor_names
    
    if sub_pre:
        orn_frs_pre = df_orn_activity.loc[:, df_orn_activity.columns.str.contains('pre')].values
        df_orn_frs -= orn_frs_pre
        upn_frs_pre = df_upn_activity.loc[:, df_upn_activity.columns.str.contains('pre')].values
        df_upn_frs -= upn_frs_pre
    
    if olf_only:
        df_neur_ids = sim.df_neur_ids.copy()
        thermo_hygro_glomeruli = np.array(['VP1d', 'VP1l', 'VP1m', 'VP2', 'VP3', 'VP4', 'VP5'])
        df_char_olf_PNs = df_neur_ids[(df_neur_ids.altype2=='uPN') & (~df_neur_ids.glom.isin(thermo_hygro_glomeruli))]
        df_char_olf_PNs_pos = df_char_olf_PNs.index.values
        neur_names_olf_PNs = df_AL_activity.iloc[df_char_olf_PNs_pos]['neur_name'].values
        df_upn_frs = df_upn_frs.loc[neur_names_olf_PNs]    
    
    max_fr = max(np.max(np.max(df_orn_frs)), np.max(np.max(df_upn_frs)))
    
    return df_orn_frs, df_upn_frs, max_fr


def make_df_AL_activity_long(df_AL_activity):
    cs = df_AL_activity.columns
    cs = cs[cs != 'glom']

    df_AL_activity_long = (df_AL_activity[cs]
           .melt(id_vars=['neur_name', 'neur_type'],
                 var_name='timeframe', 
                 value_name='fr')
        )
    df_AL_activity_long['dur_odor'] = df_AL_activity_long['timeframe'].str.contains('dur')
    df_AL_activity_long['odor_num'] = [int(n[-1]) for n in df_AL_activity_long['timeframe']]

    # add glom info
    df_AL_activity_long['glom'] = [ r.split('_')[1] if not r.startswith('LN') else np.nan for r in df_AL_activity_long.neur_name]
    df_AL_activity['glom'] = [ r.split('_')[1] if not r.startswith('LN') else np.nan for r in df_AL_activity.neur_name]
    return df_AL_activity_long

def make_glomerular_odor_responses(df_orn_frs, df_upn_frs, df_AL_activity):
    
    df_orn_glom_onoff = (df_orn_frs
     .merge(df_AL_activity[['neur_name', 'glom']], 
            left_on='neur_name', right_on='neur_name') # merge to get glomerulus info
     .drop(columns=['neur_name']) # drop neur_name
     .groupby('glom').mean()
    )
    
    df_upn_glom_onoff = (df_upn_frs
     .merge(df_AL_activity[['neur_name', 'glom']], 
            left_on='neur_name', right_on='neur_name') # merge to get glomerulus info
     .drop(columns=['neur_name']) # drop neur_name
     .groupby('glom').mean()
    )
    
    return df_orn_glom_onoff, df_upn_glom_onoff

def make_orn_upn_frs(df_AL_activity, odor_names, df_neur_ids,
                     sub_pre=True, olf_only=True, use_pre=False):
    
    df_orn_activity = df_AL_activity.loc[df_AL_activity.neur_type == 'ORN'].set_index('neur_name')
    curcol = 'pre' if use_pre else 'dur'
    df_orn_frs = df_orn_activity.loc[:, df_orn_activity.columns.str.contains(curcol)]
    df_orn_frs.columns = odor_names

    df_upn_activity = df_AL_activity.loc[df_AL_activity.neur_type == 'uPN'].set_index('neur_name')
    df_upn_frs = df_upn_activity.loc[:, df_upn_activity.columns.str.contains(curcol)]
    df_upn_frs.columns = odor_names
    
    
    if sub_pre:
        orn_frs_pre = df_orn_activity.loc[:, df_orn_activity.columns.str.contains('pre')].values
        df_orn_frs -= orn_frs_pre
        upn_frs_pre = df_upn_activity.loc[:, df_upn_activity.columns.str.contains('pre')].values
        df_upn_frs -= upn_frs_pre
    
    if olf_only:
        #df_neur_ids = sim.df_neur_ids.copy()
        thermo_hygro_glomeruli = np.array(['VP1d', 'VP1l', 'VP1m', 'VP2', 'VP3', 'VP4', 'VP5'])
        df_char_olf_PNs = df_neur_ids[(df_neur_ids.altype=='uPN') & (~df_neur_ids.glom.isin(thermo_hygro_glomeruli))]
        df_char_olf_PNs_pos = df_char_olf_PNs.index.values
        neur_names_olf_PNs = df_AL_activity.iloc[df_char_olf_PNs_pos]['neur_name'].values
        df_upn_frs = df_upn_frs.loc[neur_names_olf_PNs]    
        
    return df_orn_frs, df_upn_frs



def set_connect_from_scale_dic(custom_scale_dic, sim):
    
    keys = ['otoo', 'otol', 'otoi', 'otoe', 'otop',
            'ltoo', 'ltol', 'ltop',
            'itoo', 'itoi', 'itoe', 'itop',
            'etoo', 'etoi', 'etoe', 'etop',
            'ptoo', 'ptol', 'ptoi', 'ptoe', 'ptop',
            'ALL']
    scale_dic = {k: 1 for k in keys}
    
    for k in custom_scale_dic.keys():
        scale_dic[k] = custom_scale_dic[k]
    
    con = sim.al_block_df.copy()
        
    ornpos = sim.ORNpos; lnpos = sim.LNpos; pnpos = sim.PNpos
    ilnpos = sim.iLNpos; elnpos = sim.eLNpos    
        
    # ORNs presynaptic
    con.iloc[ornpos, ornpos] *= scale_dic['otoo']
    con.iloc[ornpos, lnpos] *= scale_dic['otol']
    if len(elnpos) > 0:
        con.iloc[ornpos, ilnpos] *= scale_dic['otoi']
        con.iloc[ornpos, elnpos] *= scale_dic['otoe']
    con.iloc[ornpos, pnpos] *= scale_dic['otop']
        
    # LNs presynaptic
    con.iloc[lnpos, ornpos] *= scale_dic['ltoo']
    con.iloc[lnpos, lnpos] *= scale_dic['ltol']
    con.iloc[lnpos, pnpos] *= scale_dic['ltop']
    
    if len(elnpos) > 0:
        # iLNs presynaptic
        con.iloc[ilnpos, ornpos] *= scale_dic['itoo']
        con.iloc[ilnpos, ilnpos] *= scale_dic['itoi']
        con.iloc[ilnpos, elnpos] *= scale_dic['itoe']
        con.iloc[ilnpos, pnpos] *= scale_dic['itop']
        
        # eLNs presynaptic
        con.iloc[elnpos, ornpos] *= scale_dic['etoo']
        con.iloc[elnpos, ilnpos] *= scale_dic['etoi']
        con.iloc[elnpos, elnpos] *= scale_dic['etoe']
        con.iloc[elnpos, pnpos] *= scale_dic['etop']

    # PNs presynaptic
    con.iloc[pnpos, ornpos] *= scale_dic['ptoo']
    con.iloc[pnpos, lnpos] *= scale_dic['ptol']
    if len(elnpos) > 0:
        con.iloc[pnpos, ilnpos] *= scale_dic['ptoi']
        con.iloc[pnpos, elnpos] *= scale_dic['ptoe']
    con.iloc[pnpos, pnpos] *= scale_dic['ptop']
    
    # scale all to set # mini PSCs / spike/ synapse
    con.iloc[:, :] *= scale_dic['ALL']
    
    return con


def make_d_neur_inputs(sim, condf):
    
    df_sim_neurons = sim.df_neur_ids.copy()
    df_sim_neurons['sim_name'] = sim.neur_names
    df_sim_neurons['sim_type'] = sim.neur_types
    
    df_orns = df_sim_neurons[df_sim_neurons.altype=='ORN']
    df_lns = df_sim_neurons[df_sim_neurons.altype=='LN']
    df_ilns = df_sim_neurons[df_sim_neurons.sim_type=='iLN']
    df_elns = df_sim_neurons[df_sim_neurons.sim_type=='eLN']
    df_pns = df_sim_neurons[df_sim_neurons.altype=='PN']
    
    pos_orn = df_orns.index.values
    pos_ln = df_lns.index.values
    pos_iln = df_ilns.index.values
    pos_eln = df_elns.index.values
    pos_pn = df_pns.index.values
    
    d_neur_inputs = {}

    for neur_i in range(sim.nAL):
        d_neur_inputs[neur_i] = {}
        neur_inputs = condf.iloc[:, neur_i].values


        neur_type = df_sim_neurons.iloc[neur_i]['sim_type']
        d_neur_inputs[neur_i]['neur_type'] = neur_type
        neur_glom = df_sim_neurons.iloc[neur_i]['glom']
        d_neur_inputs[neur_i]['neur_glom'] = np.nan

        d_neur_inputs[neur_i]['neur_inputs'] = neur_inputs

        d_neur_inputs[neur_i]['inputs_ORN'] = neur_inputs[pos_orn]
        d_neur_inputs[neur_i]['inputs_ORN_sameglom'] = np.array([])
        d_neur_inputs[neur_i]['inputs_ORN_diffglom'] = np.array([])
        d_neur_inputs[neur_i]['inputs_LN'] = neur_inputs[pos_ln]
        d_neur_inputs[neur_i]['inputs_iLN'] = neur_inputs[pos_iln]
        d_neur_inputs[neur_i]['inputs_eLN'] = neur_inputs[pos_eln]
        d_neur_inputs[neur_i]['inputs_PN'] = neur_inputs[pos_orn]
        d_neur_inputs[neur_i]['inputs_PN_sameglom'] = np.array([])
        d_neur_inputs[neur_i]['inputs_PN_diffglom'] = np.array([])


        if neur_type in ['ORN', 'PN']:
            d_neur_inputs[neur_i]['neur_glom'] = neur_glom
            pos_glom_orn = df_orns[df_orns.glom == neur_glom].index.values
            pos_non_glom_orn = pos_orn[~np.isin(pos_orn, pos_glom_orn)]
            pos_glom_pn = df_pns[df_pns.glom == neur_glom].index.values
            pos_non_glom_pn = pos_pn[~np.isin(pos_pn, pos_glom_pn)]
            d_neur_inputs[neur_i]['inputs_ORN_sameglom'] = neur_inputs[pos_glom_orn]
            d_neur_inputs[neur_i]['inputs_ORN_diffglom'] = neur_inputs[pos_non_glom_orn]
            d_neur_inputs[neur_i]['inputs_PN_sameglom'] = neur_inputs[pos_glom_pn]
            d_neur_inputs[neur_i]['inputs_PN_diffglom'] = neur_inputs[pos_non_glom_pn]

    return d_neur_inputs

def get_sim_neur_input_arrays(d_neur_inputs):
    return pd.DataFrame.from_dict(d_neur_inputs).T

def get_sim_neur_input_sums(df_sim_neurs_input_arrays):
    df_sim_neurs_input_sums = df_sim_neurs_input_arrays.copy()
    #df_sim_neurs_input_nums = df_sim_neurs_input_arrays.copy()
    cs = df_sim_neurs_input_arrays.columns[2:]
    for c in cs:
        df_sim_neurs_input_sums.loc[:, c] = [np.sum(x) for x in df_sim_neurs_input_arrays[c]]
        #df_sim_neurs_input_nums.loc[:, c] = [len(x) for x in df_sim_neurs_input_arrays[c]]
        
    #df_sim_neurs_input_means = df_sim_neurs_input_sums.copy()
    #df_sim_neurs_input_means.loc[:, cs] = df_sim_neurs_input_sums[cs].div(df_sim_neurs_input_nums[cs])
    
    return df_sim_neurs_input_sums

'''
def set_hemi_connect_from_scale_dic(scale_dic, ignore_lns=False):
    
    # read in hemibrain info
    al_df = al_block.copy()
    al_df.columns = al_df.columns.astype(np.int64).values

    al_orns = df_neur_ids[df_neur_ids.altype=='ORN']['bodyId'].values
    al_lns = df_neur_ids[df_neur_ids.altype=='LN']['bodyId'].values
    al_pns = df_neur_ids[df_neur_ids.altype=='PN']['bodyId'].values

    al_df.loc[al_orns, al_orns] *= scale_dic['otoo']
    al_df.loc[al_orns, al_lns] *= scale_dic['otol']
    al_df.loc[al_orns, al_pns] *= scale_dic['otop']

    al_df.loc[al_lns, al_orns] *= scale_dic['ltoo']
    al_df.loc[al_lns, al_lns] *= scale_dic['ltol']
    al_df.loc[al_lns, al_pns] *= scale_dic['ltop']

    al_df.loc[al_pns, al_orns] *= scale_dic['ptoo']
    al_df.loc[al_pns, al_lns] *= scale_dic['ptol']
    al_df.loc[al_pns, al_pns] *= scale_dic['ptop']
    
    return al_df.values
'''  
