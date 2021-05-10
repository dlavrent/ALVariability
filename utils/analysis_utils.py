# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 14:42:16 2021

@author: dB
"""

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from scipy.spatial.distance import pdist


file_path = os.path.abspath(__file__)
project_dir = os.path.join(file_path.split('ALVariability')[0], 'ALVariability')

sys.path.append(project_dir)

import pandas as pd

df_neur_ids = pd.read_csv(os.path.join(project_dir, 'connectomics/hemibrain_v1_2/df_neur_ids.csv'), index_col=0)
al_block = pd.read_csv(os.path.join(project_dir, 'connectomics/hemibrain_v1_2/AL_block.csv'), index_col=0)
imput_table = pd.read_csv(os.path.join(project_dir, 'odor_imputation/df_odor_door_all_odors_imput_ALS.csv'), index_col=0)


def do_PCA(odor_by_glom_table):
    # column-center the data
    odor_by_glom_table_centered = odor_by_glom_table - odor_by_glom_table.mean(0)
    pca = PCA()
    pca.fit(odor_by_glom_table_centered)
    pca_projections = pca.transform(odor_by_glom_table_centered)[:, :2]
    return pca_projections, pca
   
from utils.data_utils import make_df_AL_activity_long, make_orn_upn_frs, make_glomerular_odor_responses
from utils.plot_utils_EXTRA import plot_ornpn_hist
from utils.plot_utils import set_font_sizes

set_font_sizes()

odor_names = np.array(['benzaldehyde', 
                        'butyric acid',
                        '2,3-butanedione',
                        '1-butanol',
                        'cyclohexanone',
                        'Z3-hexenol', # originally 'cis-3-hexen-1-ol',
                        'ethyl butyrate',
                        'ethyl acetate',
                        'geranyl acetate',
                        'isopentyl acetate', # originally 'isoamyl acetate',
                        '4-methylphenol', # originally '4-methyl phenol',
                        'methyl salicylate',
                        '3-methylthio-1-propanol',
                        'octanal',
                        '2-octanone',
                        'pentyl acetate', 
                        'E2-hexenal', # originally 'trans-2-hexenal',
                        'gamma-valerolactone'])



bhand_gloms = ['DL1', 'DM1', 'DM2', 'DM3', 'DM4', 'VA2', 'VM2']
model_bhand_gloms = ['DL1', 'DM1', 'DM2', 'DM3', 'DM4', 'VA2']

df_neur_ids_bhand  = df_neur_ids.copy()[
        ((df_neur_ids.altype == 'ORN') & (df_neur_ids.glom.isin(bhand_gloms))) | 
         (df_neur_ids.altype == 'LN') | 
         ((df_neur_ids.altype == 'uPN') & (df_neur_ids.glom.isin(bhand_gloms))) |
         (df_neur_ids.altype == 'mPN')
     ]

#plot_dir = 'plot_here'
#if not os.path.exists(plot_dir):
#    os.makedirs(plot_dir)
    
    
#df_AL_activity = pd.read_csv('C:/Users/dB/deBivort/projects/ALVariability/run_model/save_sims_sensitivity_sweep/2021_4_22-5_29_59__0v12_all0.1_ecol0.45_icol0.8_pcol6.0_sweep_Bhandawat_odors_5_29_59/df_AL_activity.csv', index_col=0)    

#df_AL_activity = pd.read_csv('C:/Users/dB/deBivort/projects/ALVariability/candidates/df_AL_activity_a0.1_e0.25_i0.2_p6.0.csv', index_col=0)    


df_bhand_frs = pd.read_csv('../datasets/Bhandawat2007/fig3_responses/fig3_firing_rates.csv')
df_bhand_orn_glom_by_odor = df_bhand_frs[df_bhand_frs.cell_type == 'ORN'].pivot('glomerulus', 'odor', 'firing_rate').loc[bhand_gloms, odor_names]
df_bhand_pn_glom_by_odor = df_bhand_frs[df_bhand_frs.cell_type == 'PN'].pivot('glomerulus', 'odor', 'firing_rate').loc[bhand_gloms, odor_names]

bhand_ORN_projections, bhand_ORN_pca = do_PCA(df_bhand_orn_glom_by_odor.T)
bhand_PN_projections, bhand_PN_pca = do_PCA(df_bhand_pn_glom_by_odor.T)
    

def plot_PN_vs_ORN_comparison_to_Bhandawat(fig, axs, orn_table, pn_table):

    
    for g in bhand_gloms:
        axs[0].plot(orn_table.loc[g], pn_table.loc[g], 'o', label=g)

        axs[1].plot(df_bhand_orn_glom_by_odor.loc[g], 
                    df_bhand_pn_glom_by_odor.loc[g], 'o', label=g)

    axs[0].set_title('model')
    axs[1].set_title('Bhandawat (2007)')
    axs[0].set_xlabel('ORN firing rate (Hz)')
    axs[0].set_ylabel('PN firing rate (Hz)')
    plt.legend(title='glomerulus', loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0)
    #.show()
    
    
def plot_PCA_comparison_Bhandawat(fig, axs, dfORNpca, dfPNpca):
    

    for i in range(len(odor_names)):
        axs[0, 0].scatter(bhand_ORN_projections[i, 1], -bhand_ORN_projections[i, 0], label=odor_names[i])
        axs[0, 1].scatter(bhand_PN_projections[i, 1], -bhand_PN_projections[i, 0], label=odor_names[i])
        axs[1, 0].scatter(dfORNpca[i, 1], -dfORNpca[i, 0], label=odor_names[i])
        axs[1, 1].scatter(dfPNpca[i, 1], -dfPNpca[i, 0], label=odor_names[i])


    axs[0, 0].set_title('ORNs (Bhandawat 2007)')
    axs[0, 1].set_title('PNs (Bhandawat 2007)')    
    axs[1, 0].set_title('ORNs (model)')
    axs[1, 1].set_title('PNs (model)')
    axs[1, 0].set_xlabel('projection onto PC 2')
    axs[1, 0].set_ylabel('projection onto PC 1')


    for axset in axs:
        for ax in axset:
            ax.axvline(0, ls='--', color='0.5')
            ax.axhline(0, ls='--', color='0.5')

    #plt.show()
    
def plot_PCA_dist_comparison_Bhandawat(fig, axs, model_pca_orndists, model_pca_pndists):
    
    bhandawat_pca_orndists = pdist(bhand_ORN_projections, metric='euclidean')
    bhandawat_pca_pndists = pdist(bhand_PN_projections, metric='euclidean')
        
    arrs = [model_pca_orndists, model_pca_pndists, bhandawat_pca_orndists, bhandawat_pca_pndists]
    mindist = min([min(r) for r in arrs])
    maxdist = max([max(r) for r in arrs])
    b = np.linspace(mindist, maxdist, 36)
    
    axs[0].hist(bhandawat_pca_orndists, alpha=0.5, bins=b, label='ORN', color='xkcd:light blue')
    axs[0].hist(bhandawat_pca_pndists, alpha=0.5, bins=b, label='PN', color='xkcd:navy')
    
    axs[1].hist(model_pca_orndists, alpha=0.5, bins=b, label='ORN', color='xkcd:light blue')
    axs[1].hist(model_pca_pndists, alpha=0.5, bins=b, label='PN', color='xkcd:navy')
    
    axs[0].legend()
    axs[0].set_title('Bhandawat 2007')
    axs[1].set_title('model')
    axs[1].set_xlabel('pairwise distances between PCA-projected odors')
    #plt.show()
    

from PyPDF2 import PdfFileWriter, PdfFileReader
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch

def make_comparison_plots(df_AL_activity, plot_dir):
      
    ### BHANDAWAT VERSION
    df_AL_activity_bhand  = df_AL_activity.copy()[
        ((df_AL_activity.neur_type == 'ORN') & (df_AL_activity.glom.isin(bhand_gloms))) | 
        (df_AL_activity.neur_type.isin(['iLN', 'eLN'])) | 
        ((df_AL_activity.neur_type == 'uPN') & (df_AL_activity.glom.isin(bhand_gloms))) |
        (df_AL_activity.neur_type == 'mPN')
    ]
        
    
    ## ORN and PN firing rates
    df_AL_activity_long_bhand = make_df_AL_activity_long(df_AL_activity_bhand)
    df_orn_frs_bhand_ONOFF, df_upn_frs_bhand_ONOFF = make_orn_upn_frs(df_AL_activity_bhand, odor_names, df_neur_ids_bhand.reset_index(),
                                                  sub_pre=True, olf_only=True)
    df_orn_frs_bhand_ON, df_upn_frs_bhand_ON = make_orn_upn_frs(df_AL_activity_bhand, odor_names, df_neur_ids_bhand.reset_index(),
                                                  sub_pre=False, olf_only=True)
    
    
    # plot ORN, PN on-off firing rate histograms
    fig_ornpn_hist = plt.figure(figsize=(12,10))
    plot_ornpn_hist(df_AL_activity_long_bhand, 
                    df_orn_frs_bhand_ONOFF, df_upn_frs_bhand_ONOFF, savetag='hist_onoff_bhand', 
                    saveplot=1, savetodir=plot_dir, showplot=1)
    plt.close()
    
    # plot PN vs ORN firing rate relationship
    df_orn_glom_frs_bhand_ON, df_upn_glom_frs_bhand_ON = \
        make_glomerular_odor_responses(df_orn_frs_bhand_ON, df_upn_frs_bhand_ON, df_AL_activity_bhand)
        
        
    fig, axs = plt.subplots(1, 2, figsize=(10,5), sharex=True, sharey=True)
    plot_PN_vs_ORN_comparison_to_Bhandawat(fig, axs, df_orn_glom_frs_bhand_ON, df_upn_glom_frs_bhand_ON)
    plt.savefig(os.path.join(plot_dir, 'compare_PN_vs_ORN_bhand.png'), bbox_inches='tight')
    plt.show()
    
    df_orn_glom_frs_bhand_ONOFF, df_upn_glom_frs_bhand_ONOFF = \
        make_glomerular_odor_responses(df_orn_frs_bhand_ONOFF, df_upn_frs_bhand_ONOFF, df_AL_activity_bhand)
        
    df_orn_glom_frs_bhand_ONOFF = df_orn_glom_frs_bhand_ONOFF.loc[model_bhand_gloms, odor_names]
    df_upn_glom_frs_bhand_ONOFF = df_upn_glom_frs_bhand_ONOFF.loc[model_bhand_gloms, odor_names]
    
    model_ORN_projections, model_ORN_pca = do_PCA(df_orn_glom_frs_bhand_ONOFF.T)
    model_PN_projections, model_PN_pca = do_PCA(df_upn_glom_frs_bhand_ONOFF.T)

    
    print(model_ORN_pca.explained_variance_ratio_)
    print(model_PN_pca.explained_variance_ratio_)
    
    
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8, 8))
    plot_PCA_comparison_Bhandawat(fig, axs, model_ORN_projections, model_PN_projections)
    plt.savefig(os.path.join(plot_dir, 'compare_PCA_bhand.png'), bbox_inches='tight')
    plt.close()
    
    
    model_pca_orndists = pdist(model_ORN_projections, metric='euclidean')
    model_pca_pndists = pdist(model_PN_projections, metric='euclidean')
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6, 8))
    plot_PCA_dist_comparison_Bhandawat(fig, axs, model_pca_orndists, model_pca_pndists)
    plt.savefig(os.path.join(plot_dir, 'compare_PCA_dists_bhand.png'), bbox_inches='tight')
    plt.close()


def make_comparison_pdf(plot_dir, msg=''):
    
    
    output = PdfFileWriter()

    # add first page of PDF    
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=letter)
    t = can.beginText()
    t.setFont('Helvetica', 14)
    t.setTextOrigin(inch/4, 10.5*inch)
    t.textLines(msg)
    can.drawText(t)
    
    
    fig_PCA_dists = os.path.join(os.path.abspath(plot_dir), 'compare_PCA_dists_bhand.png')
    fig_PCA_projections = os.path.join(os.path.abspath(plot_dir), 'compare_PCA_bhand.png')
    fig_hist_onoff = os.path.join(os.path.abspath(plot_dir), 'hist_onoff_bhand.png')
    fig_pn_vs_orn = os.path.join(os.path.abspath(plot_dir), 'compare_PN_vs_ORN_bhand.png')
    
    can.drawImage(fig_PCA_projections,
                  0, 0, 
                  width=4*inch, height=4*inch, preserveAspectRatio=True)

    can.drawImage(fig_PCA_dists,
                  4.5*inch, 0, 
                  width=3.5*inch, height=4*inch, preserveAspectRatio=True)
    
    can.drawImage(fig_hist_onoff,
                  0*inch, 6*inch, 
                  width=5*inch, height=4*inch, preserveAspectRatio=True)
    
    can.drawImage(fig_pn_vs_orn,
                  0*inch, 3*inch, 
                  width=4*inch, height=4*inch, preserveAspectRatio=True)
    can.showPage()
    can.save()
    
    packet.seek(0)
    new_pdf_page = PdfFileReader(packet)
    output.addPage(new_pdf_page.getPage(0))
    
    pdf_basename = 'res_bhandawat_compare.pdf'
    out_pdf_fname = os.path.join(plot_dir, pdf_basename)
    with open(out_pdf_fname, "wb") as outputf:
        output.write(outputf)
        
    return out_pdf_fname