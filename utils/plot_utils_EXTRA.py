# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 18:41:53 2021

@author: dB
"""

       

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from itertools import product
from scipy.stats import pearsonr
from matplotlib.lines import Line2D

def set_font_sizes(SMALL_SIZE=14, MEDIUM_SIZE=16, LARGE_SIZE=20):
    '''
    Sets font size for matplotlib
    From: https://stackoverflow.com/a/39566040
    '''
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=LARGE_SIZE)   # fontsize of the figure title
    
    
def plot_sim_ORN_PN_firing_rates(sim, df_AL_activity, show_pts=False):
    
    #plt.figure(figsize=(6,6))
    hemi_gloms = sim.glom_names
    df_AL_activity = df_AL_activity[df_AL_activity.glom.isin(hemi_gloms)]
    df_glom_orn_pn_mean_fr = df_AL_activity.groupby(['glom', 'neur_type']).mean().reset_index()
    df_glom_orn_pn_std_fr = df_AL_activity.groupby(['glom', 'neur_type']).std().reset_index()
    
    for i in range(len(sim.odor_list)):

        # get odor
        odor_name = sim.odor_list[i][0]
        col = 'fr_dur_odor{}'.format(i)

        # get mean, sd firing rates for ORNs/PNs of each glomerulus
        x_orn = df_glom_orn_pn_mean_fr[df_glom_orn_pn_mean_fr.neur_type == 'ORN'][col]
        x_pn = df_glom_orn_pn_mean_fr[df_glom_orn_pn_mean_fr.neur_type == 'PN'][col]
        sig_orn = df_glom_orn_pn_std_fr[df_glom_orn_pn_std_fr.neur_type == 'ORN'][col]
        sig_pn = df_glom_orn_pn_std_fr[df_glom_orn_pn_std_fr.neur_type == 'PN'][col]

        # plot error bars
        plt.errorbar(x_orn, x_pn, fmt='o', xerr=sig_orn, yerr=sig_pn, label=odor_name)

        # plot pairwise all ORNs/PNs
        if show_pts:
            for g in hemi_gloms:
                g_orns = df_AL_activity[(df_AL_activity.neur_type == 'ORN') & (df_AL_activity.glom == g)][col]
                g_pns = df_AL_activity[(df_AL_activity.neur_type == 'PN') & (df_AL_activity.glom == g)][col]
                c = np.array(list(product(g_orns, g_pns)))
                plt.plot(c[:, 0], c[:, 1], 'o', color='k', alpha=0.05)

    
    #plt.axis('equal')
    plt.xlabel('ORN firing rate (Hz)'); plt.ylabel('PN firing rate (Hz)')
    #plt.legend(bbox_to_anchor=(1.05, 1.02), title=r'odors (mean $\pm$ sd)')
    #plt.show()
    
    



 




    
def plot_raster(ax, spike_arr, tdel=1):
    n_rows, n_cols = spike_arr.shape
    for i in range(n_rows):
        row = spike_arr[i, :]
        where_spike_in_row = np.where(row > 0)[0]
        ax.plot(where_spike_in_row/tdel, [i]*len(where_spike_in_row),
                '|', markersize=5, color='k')
    
    
def plot_driver_response_curves(Iin, I, V, synapse_series, sim_time):
    
    norm = mpl.colors.Normalize(vmin=0, vmax=max(synapse_series))
    cmap = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.jet)
    cmap.set_array([])

    #plt.figure(figsize=(12,8))
    
    gs = GridSpec(4, 2, height_ratios=[1,6,6,10], width_ratios = [20,1], 
                  hspace=0.2, wspace=0)
    
    # plot input current
    ax0 = plt.subplot(gs[0])
    ax0.plot(sim_time, Iin[0, :], color='k')
    max_int = max(Iin[0, :])
    ax0.text(0, max_int, r'$I_{in}$', va='top', color='k')
    ax0.axis('off')
    
    # plot upstream current
    ax_upstream_curr = plt.subplot(gs[1, 0], sharex=ax0)
    ax_upstream_curr.plot(sim_time, I[0, :] / 1e-12, color='0.5', label='current of upstream neuron')
    ax_upstream_curr.set_ylabel('current (pA)')
    ax_upstream_curr.legend()
    
    # plot upstream voltage
    ax_upstream_v = plt.subplot(gs[2, 0], sharex=ax0)
    ax_upstream_v.plot(sim_time, V[0, :] * 1000, color='k', label='voltage of upstream neuron')
    ax_upstream_v.set_ylabel('voltage (mV)')
    ax_upstream_v.legend()
    
    # plot downstream voltages
    ax_downstream_vs = plt.subplot(gs[3, 0], sharex=ax0)
    for i in range(len(synapse_series)):
        ax_downstream_vs.plot(sim_time, V[1+i, :] * 1000, 
                    label=synapse_series[i], c=cmap.to_rgba(synapse_series[i]))
    
    ax_downstream_vs.set_title('voltage of downstream neuron')
    ax_downstream_vs.set_xlabel('time (s)')
    ax_downstream_vs.set_ylabel('voltage (mV)')
    
    
    ax_cbar = plt.subplot(gs[3, 1])
    plt.colorbar(cmap, cax=ax_cbar, label='# miniPSCs / spike')
    
    for ax in [ax_upstream_curr, ax_upstream_v]:
        plt.setp(ax.get_xticklabels(), visible=False)
    
    #plt.show()
    

    
    
def plot_AL_activity_dur_pre_odors(df_AL_activity_long):
    #plt.figure(figsize=(8,3))
    
    # add expected behavior!
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
        #pd.Series(['PN_ex_pre', 'PN', 'fr_pre_odor0', 5, False, 0, 'DA1'], index=cols),
        #pd.Series(['PN_ex_dur', 'PN', 'fr_dur_odor0', 90, True, 0, 'DA1'], index=cols),
        pd.Series(['uPN_ex_pre', 'uPN', 'fr_pre_odor0', 5, False, 0, 'DA1'], index=cols),
        pd.Series(['uPN_ex_dur', 'uPN', 'fr_dur_odor0', 90, True, 0, 'DA1'], index=cols),
        #pd.Series(['mPN_ex_pre', 'mPN', 'fr_pre_odor0', 5, False, 0, 'DA1'], index=cols),
        #pd.Series(['mPN_ex_dur', 'mPN', 'fr_dur_odor0', 90, True, 0, 'DA1'], index=cols),
    ]
    df_expected_activity = pd.DataFrame(rows)
    # use appropriate values if all LNs, or mix of i/eLNs
    df_expected_activity = (df_expected_activity
            [df_expected_activity.neur_type.isin(df_AL_activity_long.neur_type)]
    )
    
    show_violin=True
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
                      color='k', alpha=0.05, 
                      jitter=True,
                      data=df_AL_activity_long)
    
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[0:2], labels[0:2], 
                   title='During odor?\n(diamonds are means)\n(stars are goals)', loc='upper left')
    plt.xlabel('')
    plt.ylabel('Firing rate (Hz)')
    #plt.show()
    

def plot_PN_ORN_fr_heatmaps(df_orn_frs, df_pn_frs, df_neur_ids, max_fr=200, saveto_dir='', showplot=False):
    
    def add_glom_changepts(glom_changepts, glom_list):
        for i in range(len(glom_changepts)-1):
            g1, g2 = glom_changepts[i:i+2]
            plt.fill_betweenx([0,1],x1=g1, x2=g2)
            plt.text((g1+g2)/2, 0., glom_list.iloc[g1], ha='center', va='top', rotation=90)
        
    fig = plt.figure(figsize=(16,4))
    plt.suptitle('ORN fr (Hz)')
    gs = GridSpec(2, 1, height_ratios=[12,1])
    ax1 = plt.subplot(gs[0])

    cbar_ax = fig.add_axes([.91, .2, .03, .6])

    sns.heatmap(df_orn_frs.T, ax=ax1, cbar_ax=cbar_ax,
                fmt='.0f', cmap='viridis', vmin=0, vmax=max_fr*1.05)#, cbar=True)
    ax1.set_label(''); ax1.set_xticks([])
    

    plt.subplot(gs[1], sharex=ax1)

    orn_gloms = df_neur_ids[df_neur_ids.altype=='ORN']['glom']
    glom_changepts = np.where(~orn_gloms.duplicated())[0]
    add_glom_changepts(glom_changepts, orn_gloms)
    
    plt.yticks([])
    plt.subplots_adjust(hspace=0)
    if len(saveto_dir) > 0:
        plt.savefig(os.path.join(saveto_dir, 'orn_fr_heatmap.png'), bbox_inches='tight', dpi=400)
    if showplot:
        plt.show()



    fig = plt.figure(figsize=(16,4))
    plt.suptitle('PN fr (Hz)')
    gs = GridSpec(2, 1, height_ratios=[12,1])
    ax1.set_label(''); ax1.set_xticks([])
    

    cbar_ax = fig.add_axes([.91, .2, .03, .6])

    sns.heatmap(df_pn_frs.T, ax=ax1, cbar_ax=cbar_ax,
                fmt='.0f', cmap='viridis', vmin=0, vmax=max_fr*1.05)#, cbar=True)
    ax1.set_label(''); ax1.set_xticks([])
    

    plt.subplot(gs[1], sharex=ax1)

    pn_gloms = df_neur_ids[df_neur_ids.altype=='PN']['glom']
    glom_changepts = np.where(~pn_gloms.duplicated())[0]
    add_glom_changepts(glom_changepts, pn_gloms)


    plt.yticks([])
    plt.subplots_adjust(hspace=0)
    if len(saveto_dir) > 0:
        plt.savefig(os.path.join(saveto_dir, 'pn_fr_heatmap.png'), bbox_inches='tight', dpi=400)
    if showplot:
        plt.show()
    
  
    
    
def match_ORN_fr_sim_to_input(sim, Spikes):
    all_iins = []
    all_orn_sim_frs = []; all_pn_sim_frs = []
    
    orn_names = sim.neur_names[:sim.nORNs]
    pn_names = sim.neur_names[-sim.nPNs:]
    
    for i in range(len(sim.odor_list)):
        r = sim.odor_list[i]
        odor_name = r[0]
        odor_start, odor_end = r[1], r[2]
        buf_start = odor_start + (odor_end - odor_start)/10
        iins = pd.Series(sim.Iin_rates[:, int(buf_start/sim.dt)], index=orn_names, name=odor_name)
        all_iins.append(iins)

        orn_frs, pn_frs = sim.get_ORN_PN_firing_rates(Spikes, i)
        all_orn_sim_frs.append(pd.Series(orn_frs, index=orn_names, name=odor_name))
        all_pn_sim_frs.append(pd.Series(pn_frs, index=pn_names, name=odor_name))

    sim_iins = pd.concat(all_iins, 1)
    sim_orn_frs = pd.concat(all_orn_sim_frs, 1)
    #sim_pn_frs = pd.concat(all_pn_sim_frs, 1)
    
    #plt.figure(figsize=(5,5))
    plt.axis('equal')
    plt.scatter(sim_iins.values.flatten(), 
                sim_orn_frs.values.flatten()
                )
    plt.plot([0, 200], [0, 200], color='k', ls='--')
    plt.xlabel('Input PSC rate (Hz)')
    plt.ylabel('ORN firing rate (Hz)')
    #plt.show()
    
    

    
def plot_timetable(sim, timetable, ylabel, ni1, ni2, ni3, ni4, neur_names=[], params_to_plot=[]):
    
    if len(neur_names) == 0:
        neur_names = sim.neur_names
        
    t = sim.time
    fig, axs = plt.subplots(figsize=(16,8), nrows=4, sharex=True)
    
    ni_s = [ni1, ni2, ni3, ni4]
    for i in range(len(ni_s)):
        neur_i = ni_s[i]
        axs[i].set_title('{}'.format(neur_names[neur_i]))
        # 
        for p in params_to_plot:
            axs[i].axhline(sim.params[p][neur_i], ls='--', color='k', alpha=0.5)
        axs[i].plot(t, timetable[neur_i, :], color='k')
    
        # plot odors
        ymin = min(timetable[neur_i, :]); ymax = max(timetable[neur_i, :])
        for row in sim.odor_list:
            odor_name, odor_start, odor_end = row
            axs[i].fill_between([odor_start, odor_end], [ymin, ymin], [ymax, ymax], label=odor_name, alpha=0.4)

    plt.subplots_adjust(hspace=0.4)  

    plt.xlabel('time (s)')
    for i in range(4):
        axs[i].set_ylabel(ylabel)
    plt.show()
    
    

def plot_scaled_con(fb, sim, vmin=0, vmax=3):
    
    ornpos = sim.ORNpos; lnpos = sim.LNpos; pnpos = sim.PNpos

    fig = plt.figure(figsize=(12,12))
    gs = GridSpec(3,3, width_ratios=[1,2,1], height_ratios=[1,2,1], wspace=0.025, hspace=0.025)

    cbar_ax = fig.add_axes([.92, .3, .03, .4])

    # ORN to ORN
    ax1 = fig.add_subplot(gs[0, 0])
    plot_mat(fb.iloc[ornpos, ornpos], ax1, cbar_ax, vmin, vmax)
    # ORN to LN
    ax2 = fig.add_subplot(gs[0, 1])
    plot_mat(fb.iloc[ornpos, lnpos], ax2, cbar_ax, vmin, vmax)
    # ORN to PN
    ax3 = fig.add_subplot(gs[0, 2])
    plot_mat(fb.iloc[ornpos, pnpos], ax3, cbar_ax, vmin, vmax)

    # LN to ORN
    ax4 = fig.add_subplot(gs[1, 0])
    plot_mat(fb.iloc[lnpos, ornpos], ax4, cbar_ax, vmin, vmax)
    # LN to LN
    ax5 = fig.add_subplot(gs[1, 1])
    plot_mat(fb.iloc[lnpos, lnpos], ax5, cbar_ax, vmin, vmax)
    # LN to PN
    ax6 = fig.add_subplot(gs[1, 2])
    plot_mat(fb.iloc[lnpos, pnpos], ax6, cbar_ax, vmin, vmax)

    # PN to ORN
    ax7 = fig.add_subplot(gs[2, 0])
    plot_mat(fb.iloc[pnpos, ornpos], ax7, cbar_ax, vmin, vmax)
    # PN to LN
    ax8 = fig.add_subplot(gs[2, 1])
    plot_mat(fb.iloc[pnpos, lnpos], ax8, cbar_ax, vmin, vmax)
    # PN to PN
    ax9 = fig.add_subplot(gs[2, 2])
    plot_mat(fb.iloc[pnpos, pnpos], ax9, cbar_ax, vmin, vmax)


    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
        ax.set_yticklabels([]); ax.set_yticks([]); ax.set_ylabel('')
        ax.set_xticklabels([]); ax.set_xticks([]); ax.set_xlabel('')

    ax1.set_ylabel('ORNs ({})'.format(len(ornpos))); ax7.set_xlabel('ORNs ({})'.format(len(ornpos)))
    ax4.set_ylabel('LNs ({})'.format(len(lnpos))); ax8.set_xlabel('LNs ({})'.format(len(lnpos)))
    ax7.set_ylabel('PNs ({})'.format(len(pnpos))); ax9.set_xlabel('PNs ({})'.format(len(pnpos)))

    #plt.savefig('hemibrain_conmat.png', dpi=400, bbox_inches='tight')
    fig.show()
    
    
 
def plot_orn_ln_pn_scaled_conmat(fig, df_neur_ids, final_AL_block):
    
    #fig = plt.figure(figsize=(16,16))
    cbar_ax = fig.add_axes([.92, .3, .03, .4])
    
    def plot_mat(mat, ax, cmap='jet'):
        sns.heatmap(np.log10(mat), ax=ax,
                cmap=cmap,
                vmin=0, vmax=3,
                cbar_kws={'label': r'$\log_{10}$ # synapses'},
                cbar_ax=cbar_ax)
    
       
    fb = final_AL_block.copy()
    fb.columns = pd.to_numeric(fb.columns)
    
    #fb[fb < 5] = 0
    
    final_orn_pos = np.where(df_neur_ids.altype == 'ORN')[0]
    final_ln_pos = np.where(df_neur_ids.altype == 'LN')[0]
    final_upn_pos = np.where(df_neur_ids.altype == 'uPN')[0]
    final_mpn_pos = np.where(df_neur_ids.altype == 'mPN')[0]
    
    neur_pos = [final_orn_pos, final_ln_pos, final_upn_pos, final_mpn_pos]
    neur_names = ['ORN', 'LN', 'uPN', 'mPN']
    n_types = len(neur_pos)
    neur_full_names = [neur_names[i]+'s ({})'.format(len(neur_pos[i])) for i in range(n_types)]
    
    p_ratios = np.ones((n_types,))
    p_ratios[1] = 2
       
    gs = GridSpec(n_types, n_types, 
                  width_ratios=p_ratios, height_ratios=p_ratios, 
                  wspace=0.025, hspace=0.025)       
    lw=3
    axs = []
    for i in range(n_types):
        ax_rows = []
        for j in range(n_types):
            # plot the heatmap
            ax = fig.add_subplot(gs[i, j])
            mat = fb.iloc[neur_pos[i], neur_pos[j]]
            plot_mat(mat, ax)
            print(neur_full_names[i], neur_full_names[j], mat.shape)
            # remove tick labels
            ax.set_yticklabels([]); ax.set_yticks([]); ax.set_ylabel('')
            ax.set_xticklabels([]); ax.set_xticks([]); ax.set_xlabel('')
            
            ax.axhline(y=0, color='k',linewidth=lw)
            ax.axhline(y=mat.shape[0], color='k',linewidth=lw)
            ax.axvline(x=0, color='k',linewidth=lw)
            ax.axvline(x=mat.shape[1], color='k',linewidth=lw)
    
            # add it to table of axes
            ax_rows.append(ax)
        axs.append(ax_rows)
        
    for i in range(n_types):
        axs[i][0].set_ylabel(neur_full_names[i])
        axs[n_types-1][i].set_xlabel(neur_full_names[i])
    
    #plt.savefig('outplotsub5.png', dpi=400, bbox_inches='tight')
    #plt.show()

def plot_con_single_neuron_inputs(neur_i, df_sim_neurs_input_arrays):
    
    row = df_sim_neurs_input_arrays.iloc[neur_i]
    glom = row['neur_glom']

    sets = ['inputs_ORN_sameglom', 'inputs_ORN_diffglom', 
            'inputs_PN_sameglom', 'inputs_PN_diffglom',
            'inputs_iLN', 'inputs_eLN']

    labs = [f'{glom} ORNs', f'non{glom} ORNs', f'{glom} PNs', f'non{glom} PNs', 'iLNs', 'eLNs']

    plt.figure()
    nsets = len(sets)
    for i in range(nsets):
        xx = row[sets[i]]
        labs[i] += ' ({})'.format(len(xx))
        plt.scatter(xx, np.random.normal(nsets-1-i, 0.1, len(xx)), alpha=0.3)

    plt.yticks(np.arange(nsets), labs[::-1])
    plt.title(f'Neuron index {neur_i}')
    plt.xlabel('# input synapses')
    plt.show()
    
def plot_con_multi_neuron_inputs(neur_set, df_sim_neurs_input_arrays):

    sets = ['inputs_ORN_sameglom', 'inputs_ORN_diffglom', 
            'inputs_PN_sameglom', 'inputs_PN_diffglom',
            'inputs_iLN', 'inputs_eLN']

    labs = ['glom ORNs', 'non-glom ORNs', 'glom PNs', 'non-glom PNs', 'iLNs', 'eLNs']

    nsets = len(sets)

    plt.figure()
    for neur_i in neur_set:
        row = df_sim_neurs_input_arrays.iloc[neur_i]
        for i in range(nsets):
            xx = row[sets[i]]
            #labs[i] += ' ({})'.format(len(xx))
            plt.scatter(xx, np.random.normal(nsets-1-i, 0.1, len(xx)), alpha=0.2)

    plt.yticks(np.arange(nsets), labs[::-1])
    #plt.title(f'Neuron index {neur_i}')
    plt.xlabel('# input synapses')
    plt.show()
    
    
def plot_glom_resmat(glom, res_mat, sim):

    df_neur_ids = sim.df_neur_ids.copy()
    glom_orn_pos = df_neur_ids[(df_neur_ids.altype=='ORN') & (df_neur_ids.glom==glom)].index.values
    glom_pn_pos = df_neur_ids[(df_neur_ids.altype=='uPN') & (df_neur_ids.glom==glom)].index.values

    pos_sets = [glom_orn_pos, sim.iLNpos, sim.eLNpos, glom_pn_pos]
    tits = ['ORNs', 'iLNs', 'eLNs', 'uPNs']
    colors = ['orange', 'blue', 'red', 'green']
    nsets = len(pos_sets)

    fig, axs = plt.subplots(4,1, sharex=True, figsize=(16,12))
    plt.suptitle('glom {}'.format(glom))
    for i in range(nsets):
        pos_set = pos_sets[i]
        for p in pos_set:
            axs[i].plot(sim.time, res_mat[p, :], color=colors[i], alpha=0.2)
        axs[i].plot(sim.time, res_mat[pos_set, :].mean(0), color='k')
        axs[i].set_title('{} {}'.format(len(pos_set), tits[i]))

    plt.xlabel('time (s)')
    #plt.show()

def plot_con_hists_sumcol(ddf, plot_col, lab=''):
    types = ['ORN', 'iLN', 'eLN', 'PN']
    colors = ['orange', 'blue', 'red', 'green']
    
    plot_min, plot_max = np.min(plot_col), np.max(plot_col)
    b = np.linspace(plot_min, plot_max, 50)
    
    fig, axs = plt.subplots(4,1, sharex=True)
    for i in range(len(types)):
        t = types[i]
        t_pos = ddf[ddf.neur_type == t].index.values
        axs[i].hist(plot_col.iloc[t_pos], bins=b, label=t, alpha=0.5, color=colors[i])
        axs[i].legend(bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0)
    plt.xlabel(lab)
    plt.show()
    
    


def plot_PN_fr_vs_nORNs(sim, df_AL_activity):
    n_orns_per_glom = (df_AL_activity[df_AL_activity.neur_type == 'ORN']
                       .groupby('glom').count()
                       ['neur_type'].rename('n_ORNs')
                       .sort_values()[::-1])
    df_pn_activity = (df_AL_activity[df_AL_activity['neur_type'] == 'PN']
                      .merge(n_orns_per_glom.to_frame(), left_on='glom', right_on='glom'))
    odor_names = [x[0] for x in sim.odor_list]
    #plt.figure(figsize=(7,7))
    for i in range(len(sim.odor_list)):
        pn_activities = df_pn_activity['fr_dur_odor{}'.format(i)]
        corr, p = pearsonr(df_pn_activity['n_ORNs'], pn_activities)
        lab = '{}:\ncorr {:.2f}, p < {:.2g}'.format(odor_names[i], corr, p)
        plt.scatter(df_pn_activity['n_ORNs'], pn_activities, alpha=.4, label=lab)
    plt.xlabel('number of ORNs in glomerulus'); plt.ylabel('glomerular PN firing rate (Hz)')
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    #plt.show()
       
def plot_eiLN_con_breakdown(fig, ln_to_pn_glom_synapses, iLN_names, eLN_names):
    
    glom_names = ln_to_pn_glom_synapses.columns
    iln_bool_df = ln_to_pn_glom_synapses.loc[iLN_names]
    eln_bool_df = ln_to_pn_glom_synapses.loc[eLN_names]
    
    #fig = plt.figure(figsize=(16,5))
    nE = eln_bool_df.shape[0]
    nI = iln_bool_df.shape[0]
    gs = fig.add_gridspec(ncols=3, nrows=2, wspace=0.05, hspace=0.25,
                          width_ratios=[4*nE/(nE+nI), 4*nI/(nE+nI), 1])

    names = ['eLNs', 'iLNs']
    tabs = [eln_bool_df, iln_bool_df]

    axs = []
    f_axs = [plt.subplot(gs[0, 2])]
    f_axs.append(plt.subplot(gs[1, 2], sharex=f_axs[0]))

    for i in range(2):
        innerv_bool_df = tabs[i]
        innerv_bool_df['rowsum'] = innerv_bool_df.sum(1)
        innerv_bool_df = innerv_bool_df.sort_values('rowsum', ascending=False)[glom_names]

        #axi = fig.add_subplot(gs[:, i])
        axi = plt.subplot(gs[:, i])
        sns.heatmap(innerv_bool_df.T, cmap='plasma', cbar=False, 
                yticklabels=i==0, xticklabels=False)
        plt.yticks(fontsize=6)
        axi.set_title('{} {}'.format(innerv_bool_df.shape[0], names[i]))
        axi.set_xlabel(''); axi.set_ylabel('')
        axs.append(axi)

        val_cts = innerv_bool_df.sum(1).value_counts()
        f_axs[i].bar(val_cts.index, val_cts, color='k')
        f_axs[i].set_title(names[i])

    axs += f_axs

    axs[0].set_ylabel('glom')
    axs[3].set_xlabel('# gloms innervated')
    axs[2].set_yticks([]);  axs[3].set_yticks([])
    plt.setp(axs[2].get_xticklabels(), visible=False)


    # label innervated vs. not
    leg_elems = [Line2D([0], [0], color='w', marker='s',
                        markerfacecolor='gold', markersize=10, label='innervated'),
                 Line2D([0], [0], color='w', marker='s',
                        markerfacecolor='navy', markersize=10, label='not innervated')]
    axs[0].legend(handles=leg_elems, ncol=2, bbox_to_anchor=(0, 0), loc='upper left')

    #plt.show()
    
    
def plot_orn_or_pn_syn_strengths(fig, axs, post_type, df_long_connect, nbins=50, do_log=False):
    
    in_types = ['ORN', 'eLN', 'iLN', 'PN']

    all_inputs = df_long_connect[(df_long_connect.altype_post == post_type)]['n_synapses']
    bins = np.linspace(min(all_inputs), max(all_inputs), nbins)


    #fig, axs = plt.subplots(4,1,sharex=True, figsize=(8,8))
    
    alph=0.6
    # plot ORN inputs
    axs[0].hist(df_long_connect[(df_long_connect.altype_pre == 'ORN') & 
                                (df_long_connect.altype_post == post_type) &
                                (df_long_connect.glom_pre == df_long_connect.glom_post) 
                               ]['n_synapses'], 
             label='same glom', alpha=alph, bins=bins, color='green')
    axs[0].hist(df_long_connect[(df_long_connect.altype_pre == 'ORN') & 
                                (df_long_connect.altype_post == post_type) &
                                (df_long_connect.glom_pre != df_long_connect.glom_post) 
                               ]['n_synapses'], 
             label='different glom', alpha=alph, bins=bins, color='aqua')
    axs[0].legend()

    # plot eLN inputs
    axs[1].hist(df_long_connect[(df_long_connect.altype_pre == 'eLN') & 
                                (df_long_connect.altype_post == post_type)
                               ]['n_synapses'], 
             alpha=alph, bins=bins, color='red')

    # plot iLN inputs
    axs[2].hist(df_long_connect[(df_long_connect.altype_pre == 'iLN') & 
                                (df_long_connect.altype_post == post_type)
                               ]['n_synapses'], 
             alpha=alph, bins=bins, color='blue')

    # plot PN inputs
    axs[3].hist(df_long_connect[(df_long_connect.altype_pre == 'PN') & 
                                (df_long_connect.altype_post == post_type) &
                                (df_long_connect.glom_pre == df_long_connect.glom_post)
                               ]['n_synapses'], 
             label='same glom', alpha=alph, bins=bins, color='darkorange')
    axs[3].hist(df_long_connect[(df_long_connect.altype_pre == 'PN') & 
                                (df_long_connect.altype_post == post_type) &
                                (df_long_connect.glom_pre != df_long_connect.glom_post) 
                                ]['n_synapses'], 
             label='different glom', alpha=alph, bins=bins, color='yellow')
    axs[3].legend()


    plt.xlabel('synapse strength')

    for i in range(4):    
        if do_log:
            axs[i].set_yscale('log')
        axs[i].set_title(in_types[i]+' to '+post_type)
        if i < 3:
            plt.setp(axs[i].get_xticklabels(), visible=False)
        
    fig.text(0.04, 0.5, '# of inputs', va='center', rotation='vertical')
    
    plt.suptitle(f'inputs to {post_type}s')
    plt.subplots_adjust(hspace=0.3)

    #plt.show()
    
def plot_iln_or_eln_syn_strengths(fig, axs, post_type, df_long_connect, nbins=50, do_log=False):
    
    in_types = ['ORN', 'eLN', 'iLN', 'PN']

    all_inputs = df_long_connect[(df_long_connect.altype_post == post_type)]['n_synapses']
    bins = np.linspace(min(all_inputs), max(all_inputs), nbins)


    #fig, axs = plt.subplots(4,1,sharex=True, figsize=(8,8))

    alph=0.6
    # plot ORN inputs
    axs[0].hist(df_long_connect[(df_long_connect.altype_pre == 'ORN') & 
                                (df_long_connect.altype_post == post_type)
                               ]['n_synapses'], 
             alpha=alph, bins=bins, color='green')

    # plot eLN inputs
    axs[1].hist(df_long_connect[(df_long_connect.altype_pre == 'eLN') & 
                                (df_long_connect.altype_post == post_type)
                               ]['n_synapses'], 
             alpha=alph, bins=bins, color='red')

    # plot iLN inputs
    axs[2].hist(df_long_connect[(df_long_connect.altype_pre == 'iLN') & 
                                (df_long_connect.altype_post == post_type)
                               ]['n_synapses'], 
             alpha=alph, bins=bins, color='blue')

    # plot PN inputs
    axs[3].hist(df_long_connect[(df_long_connect.altype_pre == 'PN') & 
                                (df_long_connect.altype_post == post_type)
                               ]['n_synapses'], 
             alpha=alph, bins=bins, color='darkorange')
    
    plt.xlabel('synapse strength')

    for i in range(4):    
        if do_log:
            axs[i].set_yscale('log')
        axs[i].set_title(in_types[i]+' to '+post_type)
        if i < 3:
            plt.setp(axs[i].get_xticklabels(), visible=False)
        
    fig.text(0.04, 0.5, '# of inputs', va='center', rotation='vertical')

    plt.suptitle(f'inputs to {post_type}s')

    #plt.show()