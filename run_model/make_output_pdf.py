import argparse
import os
import sys
file_path = os.path.abspath(__file__)

project_dir = os.path.join(file_path.split('ALVariability')[0], 'ALVariability')

sys.path.append(project_dir)

import pandas as pd
import pickle
from matplotlib.gridspec import GridSpec
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from PyPDF2 import PdfFileWriter, PdfFileReader
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
import seaborn as sns
from utils.plot_utils import plot_sim_spikes2, plot_AL_activity_dur_pre_odors,\
    plot_sim_ORN_PN_firing_rates, get_AL_psths, plot_AL_psths, set_font_sizes,\
    plot_neur_I_contribs, plot_PN_fr_vs_nORNs, plot_iln_or_eln_syn_strengths,\
    plot_orn_or_pn_syn_strengths, plot_eiLN_con_breakdown, plot_orn_ln_pn_scaled_conmat
from utils.data_utils import get_AL_activity_dfs, get_orn_pn_frs_from_df_AL_activity,\
    make_d_neur_inputs, make_orn_upn_frs, make_glomerular_odor_responses, make_df_AL_activity_long
import re
from datetime import datetime

set_font_sizes()

def plot_synapse_scale_hmap(fig_scale_hmap, sim):
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
    
    
from utils.model_params import params as default_params
        
def plot_electrophys_hmap(jdir_params_seed):
    
    params = ['V0', 'Vthr', 'Vmax', 'Vmin', 
              'Cmem', 'Rmem', 'APdur', 
              'PSCmag', 'PSCrisedur', 'PSCfalldur']
    neur_types = ['ORN', 'LN', 'PN']
    param_dict = jdir_params_seed['hemi_params']
    df_params = []; df_default_params = []
    
    for p in params:
        # job params
        row_ps = [param_dict['{}_{}'.format(p, t)] for t in neur_types]
        row = pd.Series(row_ps, index=neur_types, name=p)
        df_params.append(row)
        # default params
        default_row_ps = [default_params['{}_{}'.format(p, t)] for t in neur_types]
        default_row = pd.Series(default_row_ps, index=neur_types, name=p)
        df_default_params.append(default_row)
        
    pd.set_option('display.float_format', '{:.2g}'.format)
    df_params = pd.DataFrame(df_params)
    df_default_params = pd.DataFrame(df_default_params)
    df_dist_from_default = ((df_params - df_default_params)/df_default_params).replace([np.nan, -0], 0)
    df_dist_from_default = df_dist_from_default.replace([-np.inf, np.inf], [-100, 100])
    #df_dist_from_default = df_params - df_default_params
    
    df_output = df_params.copy()
    for t in neur_types:
        df_output[t] = df_params[t].map('{:.3g}'.format) + \
            '\n(' + df_default_params[t].map('{:.3g}'.format) + ')'

    
    #plt.figure(figsize=(6,6))
    plt.title('Electrophys params, default values in (),\ncolored by change from default')
    sns.heatmap(df_dist_from_default, annot=df_output, fmt='', # annot_kws={"size": 8},
                linewidths=1, linecolor='k',
                cmap='bwr', center=0, vmin=-1e-6, vmax=1e-6, cbar=False)
    plt.tight_layout()
    #plt.show()


def plot_fig_spikes(sim, Spikes, df_AL_activity, subsampling=15, msize=3):
    plt.suptitle('Raster plot', y=1.03, fontsize=20)
    plot_sim_spikes2(sim, Spikes, df_AL_activity, subsampling, msize)
    
def plot_ornpn_hist(df_AL_activity_long, df_orn_frs, df_upn_frs):
    
    bhand_filepath = os.path.join(hemi_dir, 'literature/bhandawat_fig6_quantification.csv')
    df_bhand = pd.read_csv(bhand_filepath)
    df_bhand_pn = df_bhand[df_bhand.cell_type == 'PN']
    df_bhand_orn = df_bhand[df_bhand.cell_type == 'ORN']
    
    #plt.figure(figsize=(12,8))
    gs = GridSpec(2,2, height_ratios=[1.8,1])
    
    ax1 = plt.subplot(gs[0, :])
    plt.title('Firing rates of ORNs/LNs/PNs when odors on/off')
    plot_AL_activity_dur_pre_odors(df_AL_activity_long)

    max_fr = df_AL_activity_long['fr'].max()
    b = np.arange(-20, max_fr, 20)
    keyword = 'peak'

    ax2 = plt.subplot(gs[1,0])
    plt.title('ORN FR during odors - FR off odors')
    cnts, left_pts = np.histogram(df_orn_frs.values.flatten(), bins=b)
    plt.bar(left_pts[:-1], cnts/sum(cnts),
            align='edge', width=20, color='k', alpha=.6, 
            label='model (full odor stimulus)')
    plt.bar(df_bhand_orn['firing_rate'].values, df_bhand_orn['fraction'].values, 
            align='edge', width=20, color='gold', alpha=.6, 
            label='Bhandawat 2007\n({})'.format(keyword))
    ax2.legend()
    plt.xlabel('ORN firing rate (Hz) relative to baseline')
    plt.ylabel('fraction of ORNs*odors')

    ax3 = plt.subplot(gs[1,1], sharey=ax2)
    plt.title('uPN FR during odors - FR off odors')
    cnts, left_pts = np.histogram(df_upn_frs.values.flatten(), bins=b)
    plt.bar(left_pts[:-1], cnts/sum(cnts), 
            align='edge', width=20, color='k', alpha=.6, 
            label='model (full odor stimulus)')
    plt.bar(df_bhand_pn['firing_rate'].values, df_bhand_pn['fraction'].values, 
            align='edge', width=20, color='gold', alpha=.6, 
            label='Bhandawat 2007\n({})'.format(keyword))
    ax3.legend()
    plt.xlabel('uPN firing rate (Hz) relative to baseline')
    plt.ylabel('fraction of PNs*odors')
    plt.subplots_adjust(hspace=0.1)
    plt.tight_layout()
    #if saveplot:
    #    plt.savefig(os.path.join(savetodir, f'{savetag}.png'))
    #if showplot:
    #    plt.show()
    #plt.close()
    
    
def plot_fig_orn_pn_stats(fig_orn_pn, orn_table, pn_table, cmap='bwr'):

    odor_names = orn_table.columns
    
    #fig_orn_pn = plt.figure(figsize=(12, 12))

    cbar_ax = fig_orn_pn.add_axes([.91, .4, .03, .4])

    gs = GridSpec(5, 3, height_ratios=[6,2,6,2,7], width_ratios=[1.5,2,1])
    
    odv = orn_table.T.values; pdv = pn_table.T.values
    maxz = max(np.max(odv), np.max(pdv)); minz = min(np.min(odv), np.min(pdv))
    orn_dists = pdist(odv, metric='euclidean'); pn_dists = pdist(pdv, metric='euclidean')
    
    mindist = min(min(orn_dists), min(pn_dists)); maxdist = max(max(orn_dists), max(pn_dists))
    

    ##### TOP HALF: HEATMAPS
    # ORN heatmap
    ax1 = plt.subplot(gs[0, :]); ax1.set_title('ORNs')
    sns.heatmap(orn_table.T, ax=ax1, cbar_ax=cbar_ax,
                fmt='.0f', cmap=cmap, center=0, vmin=minz, vmax=maxz)

    # filler
    axf = plt.subplot(gs[1, :]); axf.set_visible(False)

    # PN heatmap
    ax3 = plt.subplot(gs[2, :]); ax3.set_title('PNs')
    sns.heatmap(pn_table.T, ax=ax3, cbar_ax=cbar_ax,
                fmt='.0f', cmap=cmap, center=0, vmin=minz, vmax=maxz)

    # filler
    axf = plt.subplot(gs[3, :]); axf.set_visible(False)

    ##### BOTTOM HALF: COMPARISONS
    # plot firing rates PN vs ORN
    ax5 = plt.subplot(gs[4,0])
    for od in odor_names:
        plt.plot(orn_table[od], pn_table[od], 'o', label=od)
    plt.xlabel('ORN firing rate'); plt.ylabel('PN firing rate')
    plt.title('Mean glom PN vs. ORN frs')
    plt.legend(bbox_to_anchor=(1.05, -0.2), title=r'odors (mean $\pm$ sd)')

    # plot euclidean distances
    ax6 = plt.subplot(gs[4,1])
    plt.title('Distance b/w odor pairs in PN/ORN space')    
    b = np.linspace(mindist, maxdist, 12)  
    plt.hist(orn_dists, label='ORN', alpha=0.6, bins=b, color='xkcd:light blue')
    plt.hist(pn_dists, label='PN', alpha=0.6, bins=b, color='xkcd:navy')
    plt.xlabel('pairwise euclidean distance'); plt.ylabel('# odor pairs')
    plt.legend()

    # plot as scatter plot
    ax7 = plt.subplot(gs[4,2])
    plt.scatter(orn_dists, pn_dists, color='k')
    plt.plot([mindist, maxdist], [mindist, maxdist], color='0.5', ls='--')
    plt.xlabel('ORN pairwise distance'); plt.ylabel('PN pairwise distance')

    # final adjustments
    plt.subplots_adjust(hspace=0.15, wspace=0.25)
    
    #if saveplot:
    #    plt.savefig(os.path.join(savetodir, f'{savetag}.png'))
    #f showplot:
    #    plt.show()
    #plt.close()
    
    return orn_dists, pn_dists
    
def plot_psths(sim, Spikes):
    orn_psths, pn_psths, eln_psths, iln_psths = get_AL_psths(sim, Spikes)
    plot_AL_psths(orn_psths, pn_psths, eln_psths, iln_psths, sim, 6)
    plt.tight_layout()
    
    
    
def process_jdir(jdir, pdf_folder_name, commonfolder='', savepdfstoo=True):
    
    thermo_hygro_glomeruli = np.array(['VP1d', 'VP1l', 'VP1m', 'VP2', 'VP3', 'VP4', 'VP5'])
    
    
    # get list of files
    jdir_files = os.listdir(jdir)
    # get time tag
    jdir = os.path.abspath(jdir)
    time_tag_re = re.findall('(\d+)\_(\d+)\_(\d+)-(\d+)\_(\d+)\_(\d+)', jdir)[0]
    time_tag = datetime(*[int(t) for t in time_tag_re])
    run_tag = os.path.basename(os.path.normpath(jdir))
    spliton = '__0hemi'
    # add v11 hemibrain support
    if spliton not in jdir:
        spliton = '__0v11'
    job_tag = spliton + jdir.split(spliton)[1]
    job_tag = job_tag.strip('/')
    # get seed dict for params
    jdir_params_seed = pickle.load(open(os.path.join(jdir, 'sim_params_seed.p'), 'rb'))
    # get synapse dictionary
    job_synapse_ws = jdir_params_seed['custom_scale_dic']


    sim_output = pickle.load(open(os.path.join(jdir, 'sim_output.p'), 'rb'))
    sim = sim_output['sim']
    V = sim_output['V']
    I = sim_output['I'].toarray()
    Iin = sim_output['Iin'].toarray()
    Spikes = sim_output['Spikes'].toarray()
    odor_names = [r[0] for r in sim.odor_list]
    
    DPI = 250
    
    savepicsdir = os.path.join(jdir, pdf_folder_name)
    if not os.path.exists(savepicsdir):
        os.makedirs(savepicsdir)

    df_AL_activity, df_AL_activity_long = get_AL_activity_dfs(sim, Spikes)
    # remove non olfactory glomeruli
    df_AL_activity_long = df_AL_activity_long[~df_AL_activity_long.glom.isin(thermo_hygro_glomeruli)]
    df_char_ids = sim.df_char_ids.copy()


    df_orn_frs, df_upn_frs = make_orn_upn_frs(df_AL_activity, odor_names, df_char_ids,
                                              sub_pre=True, olf_only=True)
    df_orn_glom_onoff, df_upn_glom_onoff = make_glomerular_odor_responses(df_orn_frs, df_upn_frs, df_AL_activity)
    
    
    
    # plot firing rates
    print('plotting firing rates...')
    fname_frs_peak = os.path.join(savepicsdir, 'sim_AL_frs_peak.png')
    fig_frs_peak = plt.figure(figsize=(12,12))
    plot_ornpn_hist(df_AL_activity_long, df_orn_frs, df_upn_frs)
    plt.savefig(fname_frs_peak, bbox_inches='tight', dpi=DPI)
    if savepdfstoo:
        plt.savefig(fname_frs_peak.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
    # plot pn vs orn firing rates an
    print('plotting pn vs orn...')
    fname_orn_pn = os.path.join(savepicsdir, 'sim_orn_pn.png')
    fig_orn_pn = plt.figure(figsize=(12,12))
    orn_onoff_dists, upn_onoff_dists = plot_fig_orn_pn_stats(fig_orn_pn, 
                                                         df_orn_glom_onoff, 
                                                         df_upn_glom_onoff, 
                                                         'bwr')
    plt.savefig(fname_orn_pn, bbox_inches='tight', dpi=DPI)
    if savepdfstoo:
        plt.savefig(fname_orn_pn.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
   
    # plot Bhandawat version
    bhand_gloms = ['DL1', 'DM1', 'DM2', 'DM3', 'DM4', 'VA2']
    df_char_ids_bhand  = df_char_ids.copy()[((df_char_ids.altype == 'ORN') & (df_char_ids.glom.isin(bhand_gloms))) | 
              (df_char_ids.altype == 'LN') | 
               ((df_char_ids.altype2 == 'uPN') & (df_char_ids.glom.isin(bhand_gloms))) |
               (df_char_ids.altype2 == 'mPN')
              ]
    df_AL_activity_bhand  = df_AL_activity.copy()[((df_AL_activity.neur_type == 'ORN') & (df_AL_activity.glom.isin(bhand_gloms))) | 
          (df_AL_activity.neur_type.isin(['iLN', 'eLN'])) | 
           ((df_AL_activity.neur_type2 == 'uPN') & (df_AL_activity.glom.isin(bhand_gloms))) |
           (df_AL_activity.neur_type2 == 'mPN')
          ]
    df_AL_activity_long_bhand = make_df_AL_activity_long(df_AL_activity_bhand)
    df_orn_frs_bhand, df_upn_frs_bhand = make_orn_upn_frs(df_AL_activity_bhand, 
                                                          odor_names, 
                                                          df_char_ids_bhand.reset_index(),
                                                          sub_pre=True, olf_only=True)
    df_orn_glom_onoff_bhand, df_upn_glom_onoff_bhand = \
        make_glomerular_odor_responses(df_orn_frs_bhand, df_upn_frs_bhand, df_AL_activity_bhand)
    
    print('plotting firing rates Bhandawat...')
    fname_frs_peak_BHAND = os.path.join(savepicsdir, 'sim_AL_frs_peak_BHAND.png')
    fig_frs_peak_BHAND = plt.figure(figsize=(12,12))
    plot_ornpn_hist(df_AL_activity_long_bhand, df_orn_frs_bhand, df_upn_frs_bhand)
    plt.savefig(fname_frs_peak_BHAND, bbox_inches='tight', dpi=DPI)
    if savepdfstoo:
        plt.savefig(fname_frs_peak_BHAND.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
       
    print('plotting pn vs orn Bhandawat gloms...')
    fname_orn_pn_BHAND = os.path.join(savepicsdir, 'sim_orn_pn_BHANDgloms.png')
    fig_orn_pn_BHAND = plt.figure(figsize=(12,12))
    orn_bhand_dists, upn_bhand_dists = plot_fig_orn_pn_stats(fig_orn_pn_BHAND, 
                                                             df_orn_glom_onoff_bhand, 
                                                             df_upn_glom_onoff_bhand, 
                                                             'bwr')
    plt.savefig(fname_orn_pn_BHAND, bbox_inches='tight', dpi=DPI)
    if savepdfstoo:
        plt.savefig(fname_orn_pn_BHAND.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()

          
    # plot the raster
    print('plotting raster...')
    fname_spikes = os.path.join(savepicsdir, 'sim_raster.png')
    fig_spikes = plt.figure(figsize=(12,12))
    plot_fig_spikes(sim, Spikes, df_AL_activity)
    plt.savefig(fname_spikes, bbox_inches='tight', dpi=DPI)
    if savepdfstoo:
        plt.savefig(fname_spikes.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
    # plot connectivity
    print('plotting connectivity...')
    fname_conmat = os.path.join(savepicsdir, 'sim_scaled_con.png')
    fig_conmat = plt.figure(figsize=(16,16))
    alblock = pd.DataFrame(np.abs(sim.connect), index=df_char_ids.bodyId, columns=df_char_ids.bodyId)
    plot_orn_ln_pn_scaled_conmat(fig_conmat, df_char_ids, alblock)
    plt.savefig(fname_conmat, bbox_inches='tight', dpi=DPI)
    plt.close()

    # plot PN FR vs. glom size
    print('plotting PN FR vs. glom size...')
    fname_pn_vs_glom_size = os.path.join(savepicsdir, 'sim_PN_FR_vs_glom_size.png')
    fig_pn_fr_vs_glom_size = plt.figure(figsize=(6,6))
    plot_PN_fr_vs_nORNs(sim, df_AL_activity)
    plt.savefig(fname_pn_vs_glom_size, bbox_inches='tight', dpi=DPI)
    plt.close()
        
    # plot pn vs orn firing rates
    print('plotting psths...')
    fname_psth = os.path.join(savepicsdir, 'sim_psth.png')
    fig_psth = plt.figure(figsize=(12,12))
    plot_psths(sim, Spikes)
    plt.savefig(fname_psth, bbox_inches='tight', dpi=DPI)
    if savepdfstoo:
        plt.savefig(fname_psth.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
    # plot hmap synapse strengths
    print('plotting synapse strength heatmap...')
    fname_scale_hmap = os.path.join(savepicsdir, 'sim_scale_hmap.png')
    fig_scale_hmap = plt.figure(figsize=(4,4))
    plot_synapse_scale_hmap(fig_scale_hmap, sim)
    plt.savefig(fname_scale_hmap, bbox_inches='tight', dpi=DPI)
    b, t = plt.ylim() # discover the values for bottom and top
    plt.ylim(b+0.5, t-0.5) # update the ylim(bottom, top) values
    plt.close()
    
    # plot electrophys
    print('plotting electrophys...')
    fname_electro = os.path.join(savepicsdir, 'sim_electrophys.png')
    fig_electro = plt.figure(figsize=(6,8))
    plot_electrophys_hmap(jdir_params_seed)
    b, t = plt.ylim() # discover the values for bottom and top
    plt.ylim(b+0.5, t-0.5) # update the ylim(bottom, top) values
    plt.savefig(fname_electro, bbox_inches='tight', dpi=DPI)
    plt.close()
    
    
    # plot current contributions
    d_neur_inputs = make_d_neur_inputs(sim, pd.DataFrame(sim.connect))
        
    # plot PNs
    print('plotting PN current contributors...')
    #gloms_to_look_at = ['VC1', 'DM1', 'DP1l', 'DM6', 'VM1']
    #pns_to_look_at = ['PN_{}_0'.format(g) for g in gloms_to_look_at]
    #pn_look_poses = sim.neur_names[sim.neur_names.isin(pns_to_look_at)]    
    pns_to_look_at = ['PN_VC1_0', # the PN with the most ORN connections
                      'PN_DP1m_0',
                      'PN_DP1m_1',
                      'PN_DM1_0',
                      'PN_DP1l_0',
                      'PN_DC2_0',
                      'PN_D_0',
                      'PN_DM6_0',
                      'PN_VM1_0',
                      'PN_DA2_0']
    
    fnames_pn_contribs = []
    for i in range(len(pns_to_look_at)):
        lab = pns_to_look_at[i]
        neur_pos = sim.neur_names[sim.neur_names == lab].index[0]
        plt.figure(figsize=(12,6))
        ax_odor, ax1, ax2 = plot_neur_I_contribs(neur_pos, d_neur_inputs, sim, I, Iin, V, Spikes)
        ax_odor.set_title(lab)
        fname_cur_pn = os.path.join(savepicsdir, 'sim_neur_contrib_PN{}.png'.format(i))
        fnames_pn_contribs.append(fname_cur_pn)
        plt.savefig(fname_cur_pn, bbox_inches='tight', dpi=DPI/4)
        plt.close()
    
    # plot iLN current contributions:
    print('plotting iLN current contributors...')
    sorted_iLNpos = np.sort(sim.iLNpos)
    nilns = len(sorted_iLNpos)
    ilns_to_show_is = np.linspace(0, nilns-1, 10).astype(int)
    fnames_iln_contribs = []
    for iLN_i in ilns_to_show_is:
        neur_i = sorted_iLNpos[iLN_i]
        where_neur_i_among_all_lns = np.where(sim.LNpos == neur_i)[0][0]
        lab = 'iLN #{}, the {}-most connected LN of {}'.format(iLN_i + 1, where_neur_i_among_all_lns+1, sim.nLNs)
        plt.figure(figsize=(12,6))
        ax_odor, ax1, ax2 = plot_neur_I_contribs(neur_i, d_neur_inputs, sim, I, Iin, V, Spikes)
        ax_odor.set_title(lab)
        fname_cur_iln = os.path.join(savepicsdir, 'sim_neur_contrib_iLN{}.png'.format(iLN_i))
        fnames_iln_contribs.append(fname_cur_iln)
        plt.savefig(fname_cur_iln, bbox_inches='tight', dpi=DPI/4)
        plt.close()
    
    # plot eLN current contributions:
    sorted_eLNpos = np.sort(sim.eLNpos)
    nelns = len(sorted_eLNpos)
    fnames_eln_contribs = []
    if nelns > 0:
        print('plotting eLN current contributors...')
        elns_to_show_is = np.linspace(0, nelns-1, 10).astype(int)
        for eLN_i in elns_to_show_is:
            neur_i = sorted_eLNpos[eLN_i]
            where_neur_i_among_all_lns = np.where(sim.LNpos == neur_i)[0][0]
            lab = 'eLN #{}, the {}-most connected LN of {}'.format(eLN_i + 1, where_neur_i_among_all_lns+1, sim.nLNs)
            plt.figure(figsize=(12,6))
            ax_odor, ax1, ax2 = plot_neur_I_contribs(neur_i, d_neur_inputs, sim, I, Iin, V, Spikes)
            ax_odor.set_title(lab)
            fname_cur_eln = os.path.join(savepicsdir, 'sim_neur_contrib_eLN{}.png'.format(eLN_i))
            fnames_eln_contribs.append(fname_cur_eln)
            plt.savefig(fname_cur_eln, bbox_inches='tight', dpi=DPI/4)
            plt.close()

    # plot connectivity stuff only if eLNs exist
    if nelns > 0:
    
        df_char_ids['neur_name'] = sim.neur_names
        df_char_ids['neur_num'] = df_char_ids['neur_name'].str.extract(r'(\d+)$').astype(int) + 1
        df_char_ids['altype'] = sim.neur_types        
            
        df_connect = pd.DataFrame(sim.connect, index=sim.neur_names, columns=sim.neur_names)
        
        df_long_connect = (df_connect
                       .reset_index()
                       .melt('index', df_connect.columns)
                       .rename(columns={'index': 'pre',
                               'variable': 'post',
                               'value': 'n_synapses'})
                       .merge(df_char_ids, left_on='pre', right_on='neur_name')
                       .drop(columns=['bodyId', 'neur_name'])
                       .merge(df_char_ids, left_on='post', right_on='neur_name',
                             suffixes=['_pre', '_post'])
                       .drop(columns=['bodyId', 'neur_name'])
        )
        
        # get LN connectivity
        ln_names = df_char_ids[df_char_ids.altype.isin(['LN', 'iLN', 'eLN'])].charId.values
        
        ln_to_pn_long = df_long_connect[(df_long_connect.altype_pre.isin(['iLN', 'eLN'])) & 
                (df_long_connect.altype_post == 'PN')]
        
        ln_to_pn_glom_synapses = np.abs(ln_to_pn_long
         .groupby(['pre', 'glom_post'])
         .sum()['n_synapses']
         .reset_index()
         .pivot(index='pre', 
                columns='glom_post', 
                values='n_synapses')
         .loc[ln_names]
        ) > 0
        
        iLN_names = sim.neur_names[sim.iLNpos].values
        eLN_names = sim.neur_names[sim.eLNpos].values

        print('plotting i/eLN connectivity..')
        fname_ieln_con = os.path.join(savepicsdir, 'sim_ieln_con.png')
        fig_ieln_con = plt.figure(figsize=(16,5))
        plot_eiLN_con_breakdown(fig_ieln_con, ln_to_pn_glom_synapses, iLN_names, eLN_names)
        plt.savefig(fname_ieln_con, bbox_inches='tight', dpi=DPI)
        plt.close()
        
        
        print('plotting ORN inputs..')
        fname_orn_inputs = os.path.join(savepicsdir, 'sim_inputs_orn.png')
        fig_orn_inputs, axs = plt.subplots(4, 1, figsize=(10,10))
        plot_orn_or_pn_syn_strengths(fig_orn_inputs, axs, 'ORN', df_long_connect, nbins=100, do_log=True)
        plt.savefig(fname_orn_inputs, bbox_inches='tight', dpi=DPI)
        plt.close()
        
        print('plotting PN inputs..')
        fname_pn_inputs = os.path.join(savepicsdir, 'sim_inputs_pn.png')
        fig_pn_inputs, axs = plt.subplots(4, 1, figsize=(10,10))
        plot_orn_or_pn_syn_strengths(fig_pn_inputs, axs, 'PN', df_long_connect, nbins=100, do_log=True)
        plt.savefig(fname_pn_inputs, bbox_inches='tight', dpi=DPI)
        plt.close()
        
        print('plotting eLN inputs..')
        fname_eln_inputs = os.path.join(savepicsdir, 'sim_inputs_eln.png')
        fig_eln_inputs, axs = plt.subplots(4, 1, figsize=(10,10))
        plot_iln_or_eln_syn_strengths(fig_eln_inputs, axs, 'eLN', df_long_connect, nbins=100, do_log=True)
        plt.savefig(fname_eln_inputs, bbox_inches='tight', dpi=DPI)
        plt.close()
    
        print('plotting iLN inputs..')
        fname_iln_inputs = os.path.join(savepicsdir, 'sim_inputs_iln.png')
        fig_iln_inputs, axs = plt.subplots(4, 1, figsize=(10,10))
        plot_iln_or_eln_syn_strengths(fig_iln_inputs, axs, 'iLN', df_long_connect, nbins=100, do_log=True)
        plt.savefig(fname_iln_inputs, bbox_inches='tight', dpi=DPI)
        plt.close()
             
            
    print('writing output pdf...')
    
    # get run explanation
    with open(os.path.join(jdir, 'export_settings_copy.py'), 'r') as f:
        l = f.readlines()    
    s_run_exp_start = [s for s in l if 'run_explanation' in s][0]
    i_run_exp_start = l.index(s_run_exp_start)
    run_exp = l[i_run_exp_start:]
    run_exp_i_to_add = run_exp[1:].index("'''\n")
    run_exp = run_exp[:run_exp_i_to_add+2]
    run_exp[0] = "'''\n"
    run_explanation = ''.join(run_exp[1:-1])
        
    msg = f'''
    Goals:

        \tORN during an odor: between 0-250ish Hz, distributed ~exponentially 
        \tORN during no odor: range between 0-30 Hz, mean around 10 Hz
    
        \tLNs: between 0-60 Hz
        
        \tPN during an odor: between 0-250ish Hz, but distributed ~uniformly                                             
        \tPN during no odor: between 0-20 Hz, mean 5 Hz 
        
        \t We expect distances between odors in PN space > distances in ORN space,
        \t and a tighter dist in PNs than in ORNs
            
            
    Run name: 
    {jdir} 
    
    Run description: 
    {run_explanation}      
    '''
    
    output = PdfFileWriter()

    # add first page of PDF    
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=letter)#np.array(letter)*basewidth/letter[0])#, bottomup=0)
    #can.setFont('Helvetica-Bold', 72)
    t = can.beginText()
    t.setFont('Helvetica', 14)
    t.setTextOrigin(inch/4, 10.5*inch)#1/6*basewidth, 5/6*basewidth)
    t.textLines(msg)
    can.drawText(t)
    can.drawImage(fname_electro, 
                  0, 0, 
                  width=3.5*inch, height=4*inch, preserveAspectRatio=True)
    can.drawImage(fname_scale_hmap, 
                  4*inch, 0, 
                  width=3.5*inch, height=3.5*inch)
    can.showPage()
    can.save()
    # Move to the beginning of the StringIO buffer
    packet.seek(0)
    new_pdf_page = PdfFileReader(packet)
    output.addPage(new_pdf_page.getPage(0))
    
    
    # add second page
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=letter)
    can.drawImage(fname_spikes, 
                  0, 6*inch,
                  width=5*inch, height=5*inch, preserveAspectRatio=True)
    can.drawImage(fname_frs_peak, 
                  4.5*inch, 6*inch, 
                  width=4*inch, height=3.5*inch, preserveAspectRatio=True)
    can.drawImage(fname_orn_pn,
                  inch, 0,
                  width=6*inch, height=6*inch, preserveAspectRatio=True)
    can.showPage()
    can.save()
    packet.seek(0)
    new_pdf_page = PdfFileReader(packet)
    output.addPage(new_pdf_page.getPage(0))
    
    
    # add third page
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=letter)
    # plot PN current contribs
    t = can.beginText(); t.setTextOrigin(0, 10.5*inch); t.setFont('Helvetica', 8)
    t.textLine(r'Current inputs to PNs'); can.drawText(t)
    for i in range(len(fnames_pn_contribs)):
        can.drawImage(fnames_pn_contribs[i], 
                  0, (9.5-i)*inch,
                  width=2*inch, height=1*inch, preserveAspectRatio=True)
        
    # plot iLN current contribs
    t = can.beginText(); t.setTextOrigin(3*inch, 10.5*inch); t.setFont('Helvetica', 8)
    t.textLine(r'Current inputs to iLNs'); can.drawText(t)
    for i in range(len(fnames_iln_contribs)):
        can.drawImage(fnames_iln_contribs[i], 
                  3*inch, (9.5-i)*inch,
                  width=2*inch, height=1*inch, preserveAspectRatio=True)
        
    # plot eLN current contribs
    if len(fnames_eln_contribs) > 0:
        t = can.beginText(); t.setTextOrigin(6*inch, 10.5*inch); t.setFont('Helvetica', 8)
        t.textLine(r'Current inputs to eLNs'); can.drawText(t)
    for i in range(len(fnames_eln_contribs)):
        can.drawImage(fnames_eln_contribs[i], 
                  6*inch, (9.5-i)*inch,
                  width=2*inch, height=1*inch, preserveAspectRatio=True)
    can.showPage()
    can.save()
    packet.seek(0)
    new_pdf_page = PdfFileReader(packet)
    output.addPage(new_pdf_page.getPage(0))
    
    
    if nelns > 0:
        # add fourth page on connectivity
        packet = io.BytesIO()
        can = canvas.Canvas(packet, pagesize=letter)
        can.drawImage(fname_ieln_con, 
                      1*inch, 8*inch,
                      width=6*inch, height=3*inch, preserveAspectRatio=True)
        
        ww = 3.2; hh = 3.2
        can.drawImage(fname_orn_inputs, 
                      0.5*inch, 4.5*inch, 
                      width=ww*inch, height=hh*inch, preserveAspectRatio=True)
        can.drawImage(fname_pn_inputs, 
                      0.5*inch, 1*inch, 
                      width=ww*inch, height=hh*inch, preserveAspectRatio=True)
        
        can.drawImage(fname_eln_inputs, 
                      4*inch, 4.5*inch, 
                      width=ww*inch, height=hh*inch, preserveAspectRatio=True)
        can.drawImage(fname_iln_inputs, 
                      4*inch, 1*inch, 
                      width=ww*inch, height=hh*inch, preserveAspectRatio=True)
        
        can.showPage()
        can.save()
        packet.seek(0)
        new_pdf_page = PdfFileReader(packet)
        output.addPage(new_pdf_page.getPage(0))
    
    
    
    # save PDF
    pdf_basename = 'sim_res_plots{}.pdf'.format(job_tag)
    out_pdf_fname = os.path.join(savepicsdir, pdf_basename)
 
    # Finally, write "output" to a real file
    print(f'writing output pdf to {out_pdf_fname}...')
    with open(out_pdf_fname, "wb") as outputf:
        output.write(outputf)
        
    # optionally, write to a common folder
    if commonfolder != '':
        common_folder_pdf_fname = os.path.join(commonfolder, pdf_basename)
        print(f'writing output pdf to {common_folder_pdf_fname}...')
        with open(common_folder_pdf_fname, "wb") as outputf:
            output.write(outputf)
    
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # stoichiometries
    parser.add_argument('-j', '--jdir', type=str, default='2020-08-31_resample5/save_sims_resample5/2020_8_31-4_50_25__0v11_hemi_eLNs15of90_all01_e025_i065_p8_mxo400_6ALSodors_ORNfindecay__RESAMPLE5_4_50_25/',#2020-08-31_resample5/save_sims_resample5save_sims_v11/2020_7_29-19_51_34__0v11_hemi_eLNs15of90_all02_etop2_ptoe05_3odors_ORNdecay/',
                        help='dir named timetag__0hemi...')
    parser.add_argument('-a', '--all', type=int, default=1,
                        help='do all directories in the dir of -j?')
    parser.add_argument('-o', '--overwrite', type=int, default=0,
                        help='overwrite folder if exists?')
    parser.add_argument('-c', '--commonfolder', type=int, default=1,
                        help='write all pdfs to a common folder?')
    parser.add_argument('--cpath', type=str, default=os.path.join(project_dir, 'submissions_resampling', 'all_pdfs2'),
                        help='common folder name')
    args = parser.parse_args()
    master_jdir = args.jdir
    do_all = args.all
    do_overwrite = args.overwrite
    do_commonfolder = args.commonfolder
    
    if do_all:
        all_dirs = os.listdir(master_jdir)
        all_jdirs = [os.path.join(master_jdir, d) for d in all_dirs \
                     if os.path.isdir(os.path.join(master_jdir, d)) and '_0v11' in d]
    else:
        all_jdirs = [master_jdir]
        
    pdf_folder_name = 'output_figs4'
    
    master_pdf_folder_path = ''
    if do_commonfolder:
        master_pdf_folder_path = args.cpath
        if not os.path.exists(master_pdf_folder_path):
            os.makedirs(master_pdf_folder_path)
    
    for jdir in all_jdirs:
        print(f'processing {jdir}...')
        # get list of files
        jdir_files = os.listdir(jdir)
        if do_overwrite or pdf_folder_name not in jdir_files:
            try:
                process_jdir(jdir, pdf_folder_name, master_pdf_folder_path)
            except:
                print('oops failed')
        else:
            print(f'{pdf_folder_name} exists, not overwriting')