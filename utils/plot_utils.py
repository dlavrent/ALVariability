# -*- coding: utf-8 -*-
"""
Plotting utilities
"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import seaborn as sns

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

def autolabel(rects):
    '''
    Attach a text label above each bar in *rects*, displaying its height.
    From: https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html
    '''
    for rect in rects:
        height = rect.get_height()
        plt.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
def plot_value_counts_of_series(ser, 
                                num_label=False,
                                color='tab:blue', 
                                fraction=False):
    '''
    Given a pandas series of cateogorical values,
    makes a bar plot of the frequency of each value,
    with optional count label
    '''
    val_cts = ser.value_counts()
    if fraction:
        val_cts /= len(ser)
    rects = plt.bar(val_cts.index, val_cts, color=color)
    if num_label:
        autolabel(rects)
        

def plot_mat(mat, ax, cbar_ax, cmap='magma'):
    '''
    Helper for plot_scaled_hmap,
    plotting a single connectivity matrix (mat)

    '''
    sns.heatmap(np.log10(mat), ax=ax,
                cmap=cmap,
                vmin=0, vmax=3,
                cbar_kws={'label': r'$\log_{10}$ # synapses'},
                cbar_ax=cbar_ax)
        
def plot_scaled_hmap(fig, conmat, neur_sets, neur_set_names, cmap='jet'):
    '''
    Intended use is to plot connectivity matrix for ORNs, LNs, u/mPNs,
    adding higher visual weight to LNs
    '''
    
    n_types = len(neur_sets)
    neur_full_names = [neur_set_names[i]+'s ({})'.format(len(neur_sets[i])) for i in range(n_types)]

    p_ratios = np.ones((n_types,))
    p_ratios[1] = 2

    gs = GridSpec(n_types, n_types, 
              width_ratios=p_ratios, height_ratios=p_ratios, 
              wspace=0.025, hspace=0.025)

    plt.suptitle(r'Hemibrain connectivity matrix', y=0.93)

    cbar_ax = fig.add_axes([.92, .3, .03, .4])

    lw = 3
    axs = []
    for i in range(n_types):
        ax_rows = []
        for j in range(n_types):
            # plot the heatmap
            ax = fig.add_subplot(gs[i, j])
            mat = conmat.loc[neur_sets[i], neur_sets[j]]
            plot_mat(mat, ax, cbar_ax, cmap='jet')

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