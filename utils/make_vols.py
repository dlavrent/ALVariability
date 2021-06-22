# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 13:51:55 2021

@author: dB
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy import stats
from scipy import optimize

file_path = os.path.abspath(__file__)
project_dir = os.path.join(file_path.split('ALVariability')[0], 'ALVariability')
sys.path.append(project_dir)

df_neur_ids = pd.read_csv(os.path.join(project_dir, 'connectomics/hemibrain_v1_2/df_neur_ids.csv'), index_col=0)
al_block = pd.read_csv(os.path.join(project_dir, 'connectomics/hemibrain_v1_2/AL_block.csv'), index_col=0)
al_block.columns = al_block.columns.astype(np.int64)
al_block.index = al_block.index.astype(np.int64)


glom_convex_hull_vols = pd.read_csv(os.path.join(project_dir, 'analysis/glomerulus_convex_hull_volumes.csv'))
glom_convex_hull_vols.columns = ['glom', 'vol']
glom_convex_hull_vols = glom_convex_hull_vols.set_index('glom')['vol']


# get IDs
orn_ids = df_neur_ids[df_neur_ids.altype == 'ORN']['bodyId'].values
ln_ids = df_neur_ids[df_neur_ids.altype == 'LN']['bodyId'].values
upn_ids = df_neur_ids[df_neur_ids.altype == 'uPN']['bodyId'].values
mpn_ids = df_neur_ids[df_neur_ids.altype == 'mPN']['bodyId'].values


hemi_gloms = df_neur_ids[df_neur_ids.altype == 'ORN']['glom'].value_counts().index.values


df_glom_inputs = []
for g in hemi_gloms:
    g_pns = df_neur_ids[(df_neur_ids.altype == 'uPN') & (df_neur_ids.glom == g)]['bodyId'].values
    g_orns = df_neur_ids[(df_neur_ids.altype == 'ORN') & (df_neur_ids.glom == g)]['bodyId'].values
    g_block_all = al_block.loc[:, g_pns].sum(0).values
    
    g_block_orn = al_block.loc[g_orns, g_pns].sum(0).values
    g_block_ln = al_block.loc[ln_ids, g_pns].sum(0).values
    g_block_same_pn = al_block.loc[g_pns, g_pns].sum(0).values
    
    
    other_orn_ids = orn_ids[~np.isin(orn_ids, g_orns)]
    
    other_upn_ids = upn_ids[~np.isin(upn_ids, g_pns)]
    
    g_block_other_orn = al_block.loc[other_orn_ids, g_pns].sum(0).values
    
    g_block_other_upn = al_block.loc[other_upn_ids, g_pns].sum(0).values
    g_block_mpn = al_block.loc[mpn_ids, g_pns].sum(0).values
    
    g_vol_hemi = glom_convex_hull_vols.loc[g]   
    
    df_glom_inputs.append(
        pd.DataFrame({
              'glom': [g]*len(g_pns),
              'pns': g_pns, 
              'ORN (same glom)': g_block_orn, 
              'ORN (diff glom)': g_block_other_orn,
              'uPN (same glom)': g_block_same_pn,
              'uPN (diff glom)': g_block_other_upn,
              'mPN': g_block_mpn,
              'LN': g_block_ln,
              'all_input': g_block_all,
              'convex_hull_vol': [g_vol_hemi]*len(g_pns)}))
    
df_glom_inputs = pd.concat(df_glom_inputs)

df_glom_vols_synapses = df_glom_inputs[['glom', 'all_input', 'convex_hull_vol']].groupby('glom').agg({'all_input': ['sum'], 
                                                                                          'convex_hull_vol': ['mean']})

Vs = df_glom_vols_synapses['convex_hull_vol']['mean'].values
Ss = df_glom_vols_synapses['all_input']['sum'].values

def ll_lognormal_exp(params, Vs, Ss):
    a, sd, d = params
    return -np.sum(stats.lognorm.logpdf(Ss, s=sd, scale=a*Vs**d))

max_lik_res = optimize.minimize(ll_lognormal_exp, x0=(1, 0.5, 1), args=(Vs, Ss))
a_mle, sd_mle, d_mle = max_lik_res['x']


drawn_glom_synapses = pd.Series(stats.lognorm.rvs(s=sd_mle, scale=a_mle*Vs**d_mle), index=df_glom_vols_synapses.index)

def adjust_glomerular_synapses_AL_block(df_neur_ids_RESAMPLE, al_block_RESAMPLE):
    
    for g in hemi_gloms:
    
        g_RESAMPLE_pns = df_neur_ids_RESAMPLE[(df_neur_ids_RESAMPLE.altype == 'uPN') & (df_neur_ids_RESAMPLE.glom == g)]['bodyId'].values
    
        g_RESAMPLE_pn_inputs = al_block_RESAMPLE.loc[:, al_block_RESAMPLE.columns.isin(g_RESAMPLE_pns)]
        
        g_old_tot_synapse_counts = g_RESAMPLE_pn_inputs.sum().sum()
        g_new_tot_synapse_counts = drawn_glom_synapses.loc[g]
    
        al_block_RESAMPLE.loc[:, al_block_RESAMPLE.columns.isin(g_RESAMPLE_pns)] = np.floor(g_RESAMPLE_pn_inputs * g_new_tot_synapse_counts  / g_old_tot_synapse_counts)
    
    return al_block_RESAMPLE



def plot_comparison_cones(df_neur_ids_RESAMPLE, al_block_RESAMPLE, saveto_dir = '', showplots=0):
    
    
    orn_ids_resample = df_neur_ids_RESAMPLE[df_neur_ids_RESAMPLE.altype == 'ORN']['bodyId'].values
    ln_ids_resample = df_neur_ids_RESAMPLE[df_neur_ids_RESAMPLE.altype == 'LN']['bodyId'].values
    upn_ids_resample = df_neur_ids_RESAMPLE[df_neur_ids_RESAMPLE.altype == 'uPN']['bodyId'].values
    mpn_ids_resample = df_neur_ids_RESAMPLE[df_neur_ids_RESAMPLE.altype == 'mPN']['bodyId'].values
    
    np.random.seed(124)
    glom_colors = {g: np.random.uniform(0, 1, 3) for g in hemi_gloms}
    
    fig, axs = plt.subplots(2, 3, figsize=(16,10), sharex=True, sharey=True)
    
    df_glom_inputs_resample = []
    for g in hemi_gloms:
        
        
        g_RESAMPLE_pns = df_neur_ids_RESAMPLE[(df_neur_ids_RESAMPLE.altype == 'uPN') & (df_neur_ids_RESAMPLE.glom == g)]['bodyId'].values
        g_RESAMPLE_orns = df_neur_ids_RESAMPLE[(df_neur_ids_RESAMPLE.altype == 'ORN') & (df_neur_ids_RESAMPLE.glom == g)]['bodyId'].values
        
            
        g_RESAMPLE_block_all = al_block_RESAMPLE.loc[:,
                                        al_block_RESAMPLE.columns.isin(g_RESAMPLE_pns)
                                    ].sum(0).values
        
        
        
        
        g_RESAMPLE_block_orn = al_block_RESAMPLE.loc[
                                    al_block_RESAMPLE.columns.isin(g_RESAMPLE_orns), 
                                    al_block_RESAMPLE.columns.isin(g_RESAMPLE_pns)
                                ].sum(0).values
        g_RESAMPLE_block_ln = al_block_RESAMPLE.loc[
                                    al_block_RESAMPLE.columns.isin(ln_ids_resample), 
                                    al_block_RESAMPLE.columns.isin(g_RESAMPLE_pns)
                                ].sum(0).values
        g_RESAMPLE_block_same_pn = al_block_RESAMPLE.loc[
                                    al_block_RESAMPLE.columns.isin(g_RESAMPLE_pns), 
                                    al_block_RESAMPLE.columns.isin(g_RESAMPLE_pns)
                                ].sum(0).values
        
        
        other_orn_ids_resample = orn_ids_resample[~np.isin(orn_ids_resample, g_RESAMPLE_orns)]
        other_upn_ids_resample = upn_ids_resample[~np.isin(upn_ids_resample, g_RESAMPLE_pns)]
        
        g_RESAMPLE_block_other_orn = al_block_RESAMPLE.loc[
                                    al_block_RESAMPLE.columns.isin(other_orn_ids_resample), 
                                    al_block_RESAMPLE.columns.isin(g_RESAMPLE_pns)
                                ].sum(0).values
        g_RESAMPLE_block_other_upn = al_block_RESAMPLE.loc[
                                    al_block_RESAMPLE.columns.isin(other_upn_ids_resample), 
                                    al_block_RESAMPLE.columns.isin(g_RESAMPLE_pns)
                                ].sum(0).values
        
        g_RESAMPLE_block_mpn = al_block_RESAMPLE.loc[
                                    al_block_RESAMPLE.columns.isin(mpn_ids_resample), 
                                    al_block_RESAMPLE.columns.isin(g_RESAMPLE_pns)
                                ].sum(0).values
        
        g_vol_hemi = glom_convex_hull_vols.loc[g]   
        
        df_glom_inputs_resample.append(
            pd.DataFrame({
                  'glom': [g]*len(g_RESAMPLE_pns),
                  'pns': g_RESAMPLE_pns, 
                  'ORN (same glom)': g_RESAMPLE_block_orn, 
                  'ORN (diff glom)': g_RESAMPLE_block_other_orn,
                  'uPN (same glom)': g_RESAMPLE_block_same_pn,
                  'uPN (diff glom)': g_RESAMPLE_block_other_upn,
                  'mPN': g_RESAMPLE_block_mpn,
                  'LN': g_RESAMPLE_block_ln,
                  'all_input': g_RESAMPLE_block_all,
                  'convex_hull_vol': [g_vol_hemi]*len(g_RESAMPLE_pns)}))
        
        
        axs[0, 0].scatter(g_vol_hemi, g_RESAMPLE_block_orn.sum(), label=g, color=glom_colors[g], alpha=0.7)
        axs[0, 1].scatter(g_vol_hemi, g_RESAMPLE_block_other_orn.sum(), label=g, color=glom_colors[g], alpha=0.7)
        axs[0, 2].scatter(g_vol_hemi, g_RESAMPLE_block_ln.sum(), label=g, color=glom_colors[g], alpha=0.7)
        axs[1, 0].scatter(g_vol_hemi, g_RESAMPLE_block_same_pn.sum(), label=g, color=glom_colors[g], alpha=0.7)
        axs[1, 1].scatter(g_vol_hemi, g_RESAMPLE_block_other_upn.sum(), label=g, color=glom_colors[g], alpha=0.7)
        axs[1, 2].scatter(g_vol_hemi, g_RESAMPLE_block_mpn.sum(), label=g, color=glom_colors[g], alpha=0.7)
        
    labadds = ['same glom ORN', 'different glom ORN', 'LN', 'same glom PN', 'other glom PN', 'mPN']
    for i in range(6):
        rowpos = i // 3
        colpos = i % 3
        ax = axs[rowpos, colpos]
        ax.set_title(labadds[i])
        
    axs[1, 0].set_ylabel('synapses from [title] onto PNs')
    axs[1, 0].set_xlabel('hemibrain glom volume (um$^3$)')
    plt.savefig(os.path.join(saveto_dir, 'adjusted_PN_synapses_breakdown.png'))
    if showplots:
        plt.show()
    plt.close()
    
    
    df_glom_inputs_resample = pd.concat(df_glom_inputs_resample)
    
    
    
    dt1 = df_glom_inputs[['glom', 'all_input', 'convex_hull_vol']].groupby('glom').agg({'all_input': ['sum'], 
                                                                                              'convex_hull_vol': ['mean']})
    dt2 = df_glom_inputs_resample[['glom', 'all_input', 'convex_hull_vol']].groupby('glom').agg({'all_input': ['sum'], 
                                                                                              'convex_hull_vol': ['mean']})
    
    fig, axs = plt.subplots(1, 2, figsize=(12,5), sharex=True, sharey=True)
    axs[0].scatter(dt1['convex_hull_vol'], dt1['all_input'])
    axs[1].scatter(dt2['convex_hull_vol'], dt2['all_input'])
    
    select_gloms = ['DP1m', 'DC4', 'DM3']
    
    for sg in select_gloms:
        axs[0].scatter(dt1.loc[sg, 'convex_hull_vol'], dt1.loc[sg, 'all_input'])
        axs[1].scatter(dt2.loc[sg,'convex_hull_vol'], dt2.loc[sg, 'all_input'], label=sg)
        
    axs[1].legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0)
    
    axs[0].set_title('hemibrain connectivity')
    axs[1].set_title('resampled connectivity,\nadjusted with glomerular synapse counts')
    for ax in axs:
        ax.set_xlabel('hemibrain glom volume (um$^3$)')
    axs[0].set_ylabel('total synapses onto PNs')
    plt.savefig(os.path.join(saveto_dir, 'compare_PN_synapses.png'))
    if showplots:
        plt.show()
    plt.close()



if 0:
        
    resample_dir = '2021_5_19-16_30_3__0v12_all0.1_ecol0.4_icol0.2_pcol4_resample_ORN_LN_umPN__16_30_3/'
    
    df_neur_ids_RESAMPLE = pd.read_csv(os.path.join(project_dir, 'run_model/save_sims_resampling_ORNs_LNs_PNs/', 
                                                    resample_dir, 'df_neur_ids.csv'), index_col=0)
    
    al_block_adjusted = adjust_glomerular_synapses_AL_block(df_neur_ids_RESAMPLE, al_block)
    plot_comparison_cones(df_neur_ids_RESAMPLE, al_block_adjusted, saveto_dir = '', showplots=1)