# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 18:10:23 2021

@author: dB
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

def plot_sim_spikes2(sim, Spikes, df_AL_activity, subsampling=15, msize=3):    
    '''
    Plots a spike raster for all neurons
    '''
    # set ordering
    df_orn_activity = df_AL_activity.copy()[df_AL_activity.neur_type == 'ORN']
    dur_cols = [c for c in df_AL_activity.columns if 'dur' in c]
    df_orn_activity['mean_dur_odors'] = df_AL_activity[dur_cols].mean(1)
    gloms_in_orn_act_order = df_orn_activity.groupby(['glom']).mean()['mean_dur_odors'].sort_values()[::-1].index.values
    df_orn_activity['glom'] = pd.Categorical(df_orn_activity['glom'], categories=gloms_in_orn_act_order, ordered=True)
    df_orns = df_orn_activity.sort_values(['glom', 'mean_dur_odors'], ascending=[1,0])
    
    ln_order = np.concatenate((sim.eLNpos, sim.iLNpos))
    
    df_pns = sim.df_neur_ids.copy()[sim.df_neur_ids.altype.isin(['uPN', 'mPN'])]
    df_pns['glom'] = pd.Categorical(df_pns['glom'], categories=gloms_in_orn_act_order, ordered=True)
    df_pns = df_pns.sort_values(['altype', 'glom'], ascending=[0,1])
    pn_order = df_pns.index.values
    
    # subsample ORNs
    orn_val_cnts = df_orn_activity.glom.value_counts()
    orn_val_cnts_subsample = np.ceil(orn_val_cnts/subsampling).astype(int)
    orn_indices_subsampled = []
    for g in gloms_in_orn_act_order:
        orn_indices = df_orns[df_orns.glom == g].index.values
        orn_n_subsamples = orn_val_cnts_subsample.loc[g]
        takespace = orn_indices[np.linspace(0, len(orn_indices)-1, orn_n_subsamples, dtype=int)]
        orn_indices_subsampled.append(takespace)
    orn_indices_subsampled = np.concatenate(orn_indices_subsampled)
    
    
    rordering = np.concatenate((orn_indices_subsampled, ln_order, pn_order))   
    
    plt.figure(figsize=(12,12))
    
    df_neur_ids = sim.df_neur_ids.copy()
    df_neur_ids.loc[sim.LNpos, 'polarity'] = -1
    df_neur_ids.loc[sim.eLNpos, 'polarity'] = +1
    df_neur_ids = df_neur_ids.iloc[rordering].reset_index()
    df_neur_ids['text_lab'] = ''
    df_neur_ids.loc[~df_neur_ids['glom'].duplicated(), 'text_lab'] = df_neur_ids.loc[~df_neur_ids['glom'].duplicated(), 'glom']
    df_neur_ids.loc[df_neur_ids.altype == 'LN', 'text_lab'] = ''
    df_neur_ids.loc[df_neur_ids.altype == 'mPN', 'text_lab'] = ''
    df_neur_ids['text_lab_pos'] = 0
    df_neur_ids.loc[df_neur_ids['text_lab'] != '', 'text_lab_pos'] = np.arange(df_neur_ids.loc[df_neur_ids['text_lab'] != ''].shape[0])
    
    
    df_neur_ids['rastype'] = df_neur_ids['altype']
    df_neur_ids.loc[(df_neur_ids.altype == 'LN') & (df_neur_ids.polarity == +1), 'rastype'] = 'eLN'
    df_neur_ids.loc[(df_neur_ids.altype == 'LN') & (df_neur_ids.polarity == -1), 'rastype'] = 'iLN'
    
    
    Spikes_subsampled = Spikes[rordering, :]
    N_CELLS = Spikes_subsampled.shape[0]
    
    gs = GridSpec(2, 2,
                  height_ratios=[1,20], width_ratios=[1,12],
                  wspace=0); gs.update(hspace=0.)
    gs.update(hspace=0.)
       
    # plot spikes
    ax2 = plt.subplot(gs[1,1])
    for i in range(N_CELLS):
        input_i = Spikes_subsampled[i, :]
        where_spike_i = np.where(input_i > 0)[0]
        ax2.plot(where_spike_i*sim.dt, [N_CELLS-i+1]*len(where_spike_i), '|', markersize=msize, color='k')
    
    # plot odors along top
    ax1 = plt.subplot(gs[0, 1], sharex=ax2)
    ax1.axis('off')
    for row in sim.odor_list:
        odor_name, odor_start, odor_end = row
        ax1.fill_between([odor_start, odor_end], [N_CELLS+1.5, N_CELLS+1.5], label=odor_name, alpha=1)
    ax1.legend(title='odors', loc='upper left', bbox_to_anchor=(1.01, 0))
    plt.xticks(size=15)
    
    
    # plot along the side
    ax3 = plt.subplot(gs[1,0], sharey=ax2)
    ax3.axis('off')
    
    
    colors_std = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    pn_gloms = df_neur_ids.loc[df_neur_ids.altype == 'uPN', 'glom'].astype(str).unique()
    glom_colors = dict((pn_gloms[i], colors_std[i % len(colors_std)]) for i in range(len(pn_gloms)))
    
    
    ntypes = ['ORN', 'eLN', 'iLN', 'uPN', 'mPN']
    for nt in ntypes:
        pos_nt = df_neur_ids[df_neur_ids.rastype == nt].index.values
        plt.text(-0.1, N_CELLS - np.mean(pos_nt), nt, ha='right')
    
    # label iLNs
    for i in range(N_CELLS):
        g1 = df_neur_ids.index[i]; g2 = g1 + 1
        neur_entry = df_neur_ids.iloc[i]
        c = 'k'
        if neur_entry.rastype == 'ORN' or neur_entry.rastype == 'uPN':
            c = glom_colors[neur_entry.glom]
        elif neur_entry.rastype == 'mPN':
            c = '0.5'
        elif neur_entry.rastype == 'eLN':
            c = 'red'
        elif neur_entry.rastype == 'iLN':
            c = 'blue'
        y1, y2 = N_CELLS-g2, N_CELLS-g1
        plt.fill_between([0,1],y1=y1, y2=y2, color=c)
        #plt.text(1/5 + 4/5*(i%5)/5, (y1+y2)/2, neur_entry.text_lab, ha='center', va='center', fontsize=6)
        plt.text((neur_entry.text_lab_pos%3)/3, (y1+y2)/2, neur_entry.text_lab, ha='left', va='center', fontsize=6)
    
    ax2.set_xlabel('Time (s)', size=15)
    ax2.set_yticks([])
    plt.tight_layout()   
    

def get_PSTH_from_spike_array(spike_array, wl=100, dt=0.0001):
    '''
    Helper for get_AL_psths to compute peristimulus time histograms
    This function returns a smoothed spike array given a desired window length
    '''
    spike_sum = spike_array.sum(0)
    n_neurs = spike_array.shape[0]
       
    resx = []
    for i in range(len(spike_sum)):
        lb = int(max(0, i-wl/2)); ub = int(min(len(spike_sum), i+wl/2))
        resx.append(spike_sum[lb:ub].sum()/(ub-lb))
        
    return np.array(resx)/n_neurs/dt

def get_AL_psths(sim, Spikes, wl=100):
    '''
    Helper for plot_AL_psths, returning the peristimulus time histograms
    for ORNs, PNs, eLNs, and uPNs
    '''
    df_neur_ids = sim.df_neur_ids.copy()
    hemi_gloms = sim.glom_names
    orn_psths = []; upn_psths = []
    for glom in hemi_gloms:
        glom_orn_pos = df_neur_ids[(df_neur_ids.altype=='ORN') & (df_neur_ids.glom==glom)].index.values
        glom_upn_pos = df_neur_ids[(df_neur_ids.altype=='uPN') & (df_neur_ids.glom==glom)].index.values

        orn_psths.append(get_PSTH_from_spike_array(Spikes[glom_orn_pos, :], dt=sim.dt, wl=wl))
        upn_psths.append(get_PSTH_from_spike_array(Spikes[glom_upn_pos, :], dt=sim.dt, wl=wl))
    orn_psths = np.array(orn_psths); upn_psths = np.array(upn_psths)
    
    eln_psths = np.zeros(len(sim.time))
    iln_psths = get_PSTH_from_spike_array(Spikes[sim.LNpos, :])
    if len(sim.eLNpos) > 0:
        eln_psths = get_PSTH_from_spike_array(Spikes[sim.eLNpos, :])
        iln_psths = get_PSTH_from_spike_array(Spikes[sim.iLNpos, :])
    
    return orn_psths, upn_psths, eln_psths, iln_psths

def plot_AL_psths(orn_psths, pn_psths, eln_psths, iln_psths, sim, topk=6):
    '''
    Plots peristimulus time histograms for select neurons
    '''
    hemi_gloms = sim.glom_names
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12,14))

    ordered_orn_gloms = hemi_gloms[np.argsort(orn_psths.sum(1))[::-1]]
    ordered_pn_gloms = hemi_gloms[np.argsort(pn_psths.sum(1))[::-1]]

    # plot ORNs
    for i in range(len(ordered_orn_gloms)):
        glom = ordered_orn_gloms[i]
        lab = glom if i < topk else ''
        hemi_i = np.where(hemi_gloms == glom)[0][0]
        ax1.plot(sim.time, orn_psths[hemi_i, :], label=lab)

    # plot PNs
    for i in range(len(ordered_pn_gloms)):
        glom = ordered_pn_gloms[i]
        lab = glom if i < topk else ''
        hemi_i = np.where(hemi_gloms == glom)[0][0]
        ax2.plot(sim.time, pn_psths[hemi_i, :], label=lab)

    # plot i, eLNs
    ax3.plot(sim.time, iln_psths, label='iLNs ({})'.format(len(sim.iLNpos)))
    ax3.plot(sim.time, eln_psths, label='eLNs ({})'.format(len(sim.eLNpos)))

    # titles, legends, axes labels
    ax1.set_title('ORNs'); ax2.set_title('PNs'); ax3.set_title('LNs')

    ax1.legend(title=f'top {topk} active', loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax2.legend(title=f'top {topk} active', loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax3.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)

    ax1.set_ylabel('firing rate (Hz)'); ax2.set_ylabel('firing rate (Hz)'); ax3.set_ylabel('firing rate (Hz)')
    ax3.set_xlabel('time (s)')
    #plt.show()
    return fig    


def plot_psths(sim, Spikes):
    '''
    Wrapper for plot_AL_psths
    '''
    orn_psths, pn_psths, eln_psths, iln_psths = get_AL_psths(sim, Spikes)
    plot_AL_psths(orn_psths, pn_psths, eln_psths, iln_psths, sim, 6)
    plt.tight_layout()

def plot_neur_I_contribs(neur_i, d_neur_inputs, sim, I, Iin, V, Spikes):
    '''
    Plots current inputs for a given neuron, broken up by cell type
    '''
    neur_inputs = d_neur_inputs[neur_i]['neur_inputs']

    neur_i_I_by_inputs = (neur_inputs * I.T).T + Iin[neur_i, :]
    neur_i_I_tot = neur_i_I_by_inputs.sum(0)
    
    neur_i_inputs_ORN = neur_i_I_by_inputs[sim.ORNpos, :]
    neur_i_inputs_iLN = neur_i_I_by_inputs[sim.iLNpos, :]
    neur_i_inputs_eLN = neur_i_I_by_inputs[sim.eLNpos, :]
    neur_i_inputs_PN = neur_i_I_by_inputs[sim.PNpos, :]
    
    #plt.figure(figsize=(12,8))
    gs = GridSpec(3,1, height_ratios=[0.2, 3, 1], hspace=0) 
    
    ax_odor = plt.subplot(gs[0])
    ax_odor.axis('off')
    for row in sim.odor_list:
        odor_name, odor_start, odor_end = row
        ax_odor.fill_between([odor_start, odor_end], [sim.nAL+1.5, sim.nAL+1.5], label=odor_name, alpha=1)
        #ax_odor.text((odor_start+odor_end)/2, 0, odor_name, ha='center', va='bottom')
    plt.xticks(size=15)
    
    ax1 = plt.subplot(gs[1], sharex=ax_odor)
    alph=.8
    #plt.title('neuron index {}: {}'.format(neur_i, sim.neur_names[neur_i]))
    plt.plot(sim.time, neur_i_I_tot/1e-12, alpha=1, color='k', label='tot')
    plt.plot(sim.time, neur_i_inputs_ORN.sum(0)/1e-12, alpha=alph, label='ORN')
    plt.plot(sim.time, neur_i_inputs_iLN.sum(0)/1e-12, alpha=alph, label='iLN')
    plt.plot(sim.time, neur_i_inputs_eLN.sum(0)/1e-12, alpha=alph, label='eLN')
    plt.plot(sim.time, neur_i_inputs_PN.sum(0)/1e-12, alpha=alph, label='PN')
    #plt.plot(sim.time, neur_i_I_tot/1e-12, alpha=alph, color='k', label='tot')
    plt.ylabel('I (pA)')
    plt.legend(loc='upper left', bbox_to_anchor=(1.01,1), borderaxespad=0)
        
    ax2 = plt.subplot(gs[2], sharex=ax1)
    ax2.plot(sim.time, V[neur_i, :]*1000, color='k')
    wherespikes = np.where(Spikes[neur_i, :] == 1)[0]
    spiketimes = sim.time[wherespikes]
    for st in spiketimes:
        plt.axvline(st, color='k', ls='-', alpha=.5)
    plt.xlabel('time (s)'); plt.ylabel('V (mV)')
        
    return ax_odor, ax1, ax2 



def plot_synapse_scale_hmap(fig_scale_hmap, sim):
    '''
    Plots a 4x4 grid indicating the cell type - cell type
    multiplier applied to the connectivity matrix of a Sim object
    (i.e. what is the multiplier applied to each iLN -> PN synapse?)
    '''
    neur_types = ['ORN', 'iLN', 'eLN', 'PN']
    neur_type_letters = ['o', 'i', 'e', 'p']
    
    original_AL_block_vals = sim.al_block_df.values
    sim_connect_vals = sim.connect
    
    divvy = np.abs(sim_connect_vals) / original_AL_block_vals
    otoovec = divvy[sim.ORNpos, :][:, sim.ORNpos].flatten(); otoivec = divvy[sim.ORNpos, :][:, sim.iLNpos].flatten(); otoevec = divvy[sim.ORNpos, :][:, sim.eLNpos].flatten(); otopvec = divvy[sim.ORNpos, :][:, sim.PNpos].flatten()
    itoovec = divvy[sim.iLNpos, :][:, sim.ORNpos].flatten(); itoivec = divvy[sim.iLNpos, :][:, sim.iLNpos].flatten(); itoevec = divvy[sim.iLNpos, :][:, sim.eLNpos].flatten(); itopvec = divvy[sim.iLNpos, :][:, sim.PNpos].flatten()
    etoovec = divvy[sim.eLNpos, :][:, sim.ORNpos].flatten(); etoivec = divvy[sim.eLNpos, :][:, sim.iLNpos].flatten(); etoevec = divvy[sim.eLNpos, :][:, sim.eLNpos].flatten(); etopvec = divvy[sim.eLNpos, :][:, sim.PNpos].flatten()
    ptoovec = divvy[sim.PNpos, :][:, sim.ORNpos].flatten(); ptoivec = divvy[sim.PNpos, :][:, sim.iLNpos].flatten(); ptoevec = divvy[sim.PNpos, :][:, sim.eLNpos].flatten(); ptopvec = divvy[sim.PNpos, :][:, sim.PNpos].flatten()
    vecs = [otoovec, otoivec, otoevec, otopvec,
            itoovec, itoivec, itoevec, itopvec,
            etoovec, etoivec, etoevec, etopvec,
            ptoovec, ptoivec, ptoevec, ptopvec ]
    vecmults = [np.unique(x[~np.isnan(x)]) for x in vecs]
    vecscales = np.array([np.abs(x[0]) for x in vecmults]).reshape(4,4)
    
    ddf = pd.DataFrame(vecscales, 
                       index=neur_type_letters, 
                       columns=neur_type_letters)
        
    ddf.columns = neur_types; ddf.index = neur_types
    ddf.index.name = 'from'; ddf.columns.name = 'to'

    plt.axis('equal')
    plt.title('multipliers on Hemibrain synapse classes\n(blank = no change)')
    mask = np.zeros_like(ddf)
    mask[ddf == 1] = True
    sns.heatmap(ddf, cmap='bwr', center=1, linewidths=1, linecolor='k',
                fmt='.3f', annot=True, cbar=False, mask=mask)
    plt.yticks(rotation=0)
    #plt.show()
    

def plot_AL_activity_dur_pre_odors(df_AL_activity_long):
    '''
    Plots firing rates for individual neurons, 
    split up by cell type, and by pre (left) / post (right) odor
    '''
    # expected firing rates
    cols = df_AL_activity_long.columns
    rows = [
        pd.Series(['ORN_ex_pre', 'ORN', 'fr_pre_odor0', 10, False, 0, 'DA1'], index=cols),
        pd.Series(['ORN_ex_dur', 'ORN', 'fr_dur_odor0', 40, True, 0, 'DA1'], index=cols),
        pd.Series(['LN_ex_pre', 'LN', 'fr_pre_odor0', 5, False, 0, np.nan], index=cols),
        pd.Series(['LN_ex_dur', 'LN', 'fr_dur_odor0', 10, True, 0, np.nan], index=cols),
        pd.Series(['iLN_ex_pre', 'iLN', 'fr_pre_odor0', 5, False, 0, np.nan], index=cols),
        pd.Series(['iLN_ex_dur', 'iLN', 'fr_dur_odor0', 10, True, 0, np.nan], index=cols),
        pd.Series(['eLN_ex_pre', 'eLN', 'fr_pre_odor0', 5, False, 0, np.nan], index=cols),
        pd.Series(['eLN_ex_dur',  'eLN', 'fr_dur_odor0', 10, True, 0, np.nan], index=cols),
        pd.Series(['uPN_ex_pre', 'uPN', 'fr_pre_odor0', 5, False, 0, 'DA1'], index=cols),
        pd.Series(['uPN_ex_dur', 'uPN', 'fr_dur_odor0', 90, True, 0, 'DA1'], index=cols),
    ]
    df_expected_activity = pd.DataFrame(rows)
    # use appropriate values if all LNs, or mix of i/eLNs
    df_expected_activity = (df_expected_activity
            [df_expected_activity.neur_type.isin(df_AL_activity_long.neur_type)]
    )
    
    show_violin=False
    show_means=True
    show_expect=True
    show_pts=True
    if show_violin:
        ax = sns.violinplot(x='neur_type', y='fr', 
                      hue='dur_odor', dodge=True, 
                      palette='husl',
                      alpha=0.1, linewidth=0,
                      data=df_AL_activity_long)
    if show_means:
        ax = sns.pointplot(x='neur_type', y='fr', 
                      hue='dur_odor', dodge=True, 
                      data=df_AL_activity_long,
                      palette="husl", join=False,
                      markers="d", ci=None, label='means')
    if show_expect:
        ax = sns.pointplot(x='neur_type', y='fr', 
                      hue='dur_odor', dodge=True, 
                      data=df_expected_activity,
                      palette=['gold']*2, join=False,
                      markers="*", ci=None, label='means')
    
    if show_pts:
        sns.stripplot(x='neur_type', y='fr', 
                      hue='dur_odor', dodge=True, 
                      color='k', alpha=0.25, size=3, 
                      jitter=True,
                      data=df_AL_activity_long)
    
    handles, labels = [], []
    plt.legend(handles, labels, frameon=False)
    plt.xlabel('')
    plt.ylabel('firing rate (Hz)')
    