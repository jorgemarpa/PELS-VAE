"""Utility functions

This file contains utility functions used during training and/or evaluation.

functions:

    * plot_wall_time_series - creates a wall plot of real an reconstrute LCs, used during model training
    * count_parameters      - return number of model's trainable parameters
    * days_hours_minutes    - return number of days, hours, and minutes from a date/time string
    * normalize_each        - normalize light curves 
    * normalize_glob        - globally normalize light curves 
    * return_dt             - return delta fimes from a sequential time array
    * plot_latent_space     - creates a figure with latent distributions during model training
    * perceptive_field      - calculates the perceptive field of a TCN net
    * str2bool              - convert Y/N string to bool
    * load_model_list       - load VAE model and config file
    * evaluate_encoder      - evaluate VAE-encoder, returns latent variables
    * plot_wall_lcs         - creates an article-ready LC plot
    * scatter_hue           - creates a color-codded scatter plot
"""

import os, re, glob
import socket
import yaml
import datetime
import torch
import numpy as np
import pandas as pd
import matplotlib
if socket.gethostname() == 'exalearn':
    matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sb
import matplotlib.cm as cm

from tqdm import tqdm_notebook
from collections import OrderedDict
from src.vae_models import *

path = os.path.dirname(os.getcwd())

# Create a wall of generated time series
def plot_wall_time_series(generated_lc, cls=[], data_real=None,
                          dim=(2, 4), figsize=(16, 4), title=None):
    """Light-curves wall plot, function used during VAE training phase.
    Figure designed and ready to be appended to W&B logger.

    Parameters
    ----------
    generated_lc : numpy array
        Array of generated light curves
    cls          : list, optional
        List of labels corresponding to the generated light curves.
    data_real    : numpy array, optional
        List of real light curves.
    dim          : list, optional
        Figure Nrows, Ncols.
    figsize      : list, optional
        Figure size
    title        : str, optional
        Figure title

    Returns
    -------
    fig
        a matplotlib figure
    image
        an image version of the figure
    """

    plt.close('all')
    if generated_lc.shape[2] == 3:
        use_time = True
        use_err = True
    elif generated_lc.shape[2] == 2:
        use_time = True
        use_err = False
    if generated_lc.shape[2] == 1:
        use_time = False
        use_err = False

    if len(cls) == 0:
        cls = [''] * (dim[0] * dim[1])
    fig, axis = plt.subplots(nrows=dim[0], ncols=dim[1], figsize=figsize)
    for i, ax in enumerate(axis.flat):
        if data_real is not None:
            ax.errorbar(data_real[i, :, 0],
                        data_real[i, :, 1],
                        yerr=data_real[i, :, 2],
                        fmt='.', c='gray', alpha=.5)
        if use_time and use_err:
            ax.errorbar(generated_lc[i, :, 0],
                        generated_lc[i, :, 1],
                        yerr=generated_lc[i, :, 2],
                        fmt='.', c='royalblue', label=cls[i])
        elif use_time and not use_err:
            ax.errorbar(generated_lc[i, :, 0],
                        generated_lc[i, :, 1], 
                        yerr=None,
                        fmt='.', c='royalblue', label=cls[i])
        elif not use_time and not use_err:
            ax.plot(generated_lc[i, :], '.',
                    c='royalblue', label=cls[i])
            
        ax.invert_yaxis()
        if cls[0] != '':
            ax.legend(loc='best')

    mytitle = fig.suptitle(title, fontsize=20, y=1.025)

    plt.tight_layout()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return fig, image


## return number of trainable parameters in the model
def count_parameters(model):
    """Calculate the number of trainable parameters of a Pytorch moel.

    Parameters
    ----------
    model : pytorh model
        Pytorch model

    Returns
    -------
    int
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


## convert time delta to days, hors and minuts
def days_hours_minutes(dt):
    """Convert ellapsed time to Days, hours, minutes, and seconds.

    Parameters
    ----------
    dt : value
        Ellapsed time

    Returns
    -------
    d
        Days
    h
        Hours
    m
        Min
    s
        Seconds
    """
    totsec = dt.total_seconds()
    d = dt.days
    h = totsec//3600
    m = (totsec%3600) // 60
    sec =(totsec%3600)%60 #just for reference
    return d, h, m, sec


## normalize light curves
def normalize_each(data, norm_time=False, scale_to=[0, 1], n_feat=3):
    """MinMax normalization of all light curves per item.

    Parameters
    ----------
    data      : numpy array
        Light curves to be normalized
    norm_time : bool array, optional
        Wheather to normalize time axis or not, default=False
    scale_to  : list, optional
        Normalize range [min, max]
    n_feat    : int, optional
        numeber of features to be normalized

    Returns
    -------
    normed
        Normalized light curves
    """
    normed = np.zeros_like(data)
    for i, lc in enumerate(data):
        for f in range(n_feat):
            normed[i, :, f] = lc[:, f]
            ## normalize time if asked
            if f == 0 and norm_time:
                normed[i, :, f] = (lc[:, f] - np.min(lc[:, f])) / \
                                  (np.max(lc[:, f]) - np.min(lc[:, f]))
            ## normalize other feature values
            if f == 1:
                normed[i, :, f] = (lc[:, f] - np.min(lc[:, f])) / \
                                  (np.max(lc[:, f]) - np.min(lc[:, f]))
            if f == 2:
                normed[i, :, f] = (lc[:, f]) / \
                                  (np.max(lc[:, f-1]) - np.min(lc[:, f-1]))
            ## scale feature values if asked
            if scale_to != [0, 1]:
                if f == 0 and not norm_time: continue
                if f == 2:
                    normed[i, :, f] = normed[i, :, f] * (scale_to[1] -
                                                         scale_to[0])
                else:
                    normed[i, :, f] = normed[i, :, f] * \
                        (scale_to[1] - scale_to[0]) + scale_to[0]
    return normed



## normalize light curves
def normalize_glob(data, norm_time=False, scale_to=[0, 1], n_feat=3):
    """MinMax normalization of all light curves with global MinMax values.

    Parameters
    ----------
    data      : numpy array
        Light curves to be normalized
    norm_time : bool array, optional
        Wheather to normalize time axis or not, default=False
    scale_to  : list, optional
        Normalize range [min, max]
    n_feat    : int, optional
        numeber of features to be normalized

    Returns
    -------
    normed
        Normalized light curves
    """
    normed = np.zeros_like(data)
    glob_min = np.min(data, axis=(0,1))
    glob_max = np.max(data, axis=(0,1))
    for i, lc in enumerate(data):
        for f in range(n_feat):
            normed[i, :, f] = lc[:, f]
            ## normalize time if asked
            if f == 0 and norm_time:
                normed[i, :, f] = (lc[:, f] - np.min(lc[:, f])) / \
                                  (np.max(lc[:, f]) - np.min(lc[:, f]))
            ## normalize other feature values
            if f == 1:
                normed[i, :, f] = (lc[:, f] - glob_min[f]) / \
                                  (glob_max[f] - glob_min[f])
            if f == 2:
                normed[i, :, f] = (lc[:, f]) / \
                                  (glob_max[f] - glob_min[f])
            ## scale feature values if asked
            if scale_to != [0, 1]:
                if f == 0 and not norm_time: continue
                if f == 2:
                    normed[i, :, f] = normed[i, :, f] * (scale_to[1] -
                                                         scale_to[0])
                else:
                    normed[i, :, f] = normed[i, :, f] * \
                        (scale_to[1] - scale_to[0]) + scale_to[0]
    return normed


## convert MJD to delta t
def return_dt(data, n_feat=3):
    """Return delta times from a sequence of observation times. 
    Time axis must be first position of last dimension

    Parameters
    ----------
    data    : numpy array
        Light curves to be processed
    n_feats : list, optional
        Number of features

    Returns
    -------
    data
        delta times
    """
    data[:,:,0] = [x-z for x, z in zip(data[:,:,0],
                                       np.min(data[:,:,0], axis=1))]
    return data


def plot_latent_space(z, y=None):
    """Creates a joint plot of features, used during training, figures
    are W&B ready

    Parameters
    ----------
    z : numpy array
        fetures to be plotted
    y : list, optional
        axis for color code

    Returns
    -------
    fig
        matplotlib figure
    fig
        image of matplotlib figure
    """
    plt.close('all')
    df = pd.DataFrame(z)
    if y is not None:
        df.loc[:,'y'] = y
    pp = sb.pairplot(df,
                     hue='y' if y is not None else None,
                     hue_order=sorted(set(y)) if y is not None else None,
                     diag_kind="hist", markers=".", height=2,
                     plot_kws=dict(s=30, edgecolors='face', alpha=.8),
                     diag_kws=dict(histtype='step'))

    plt.tight_layout()
    pp.fig.canvas.draw()
    image = np.frombuffer(pp.fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(pp.fig.canvas.get_width_height()[::-1] + (3,))
    return pp.fig, image



def perceptive_field(k=None, n=None):
    """Calculate the perceptive field of a TCN network with kernel size k
    and number of residual blocks n

    Parameters
    ----------
    k : int, opcional
        Kernel size of 1D convolutions
    n : int, optional
        Number of residual blocks 

    Returns
    -------
    pf
        perceptive field
    """
    if k != None and n != None:
        pf = 1 + 2 * (k-1) * 2**(n-1)
        print('perc_field : ', pf),
        return pf
    else:
        for k in [3,5,7,9]:
            for n in [1,2,3,4,5,6,7,8,9,10,11,12,13,14]:
                pf = 1 + 2 * (k-1) * 2**(n-1)
                if pf > 100 and pf < 400:
                    print('kernel_size: ', k)
                    print('num_blocks : ', n)
                    print('perc_field : ', pf)
                    print('######################')

                
def str2bool(v):
    """Convert strings (y,yes, true, t, 1,n, no,false, f,0) 
    to boolean values

    Parameters
    ----------
    v : numpy array
        string value to be converted to boolean

    Returns
    -------
    bool
        boolean value
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
        
def load_model_list(ID='zg3r4orb', device='cpu'):
    """Load a Python VAE model from file stored in a W&B archive

    Parameters
    ----------
    ID     : str
        W&B ID of the model to be loaded
    device : str, optional
        device where the model is loaded, cpu or gpu

    Returns
    -------
    vae
        VAE model, Python module
    conf
        Dictionary with model hyperparameters and configuration values
    """
    
    fname = glob.glob('%s/wandb/run-*-%s/VAE_model_*.pt' % (path, ID))[0]
    
    config_f = glob.glob('%s/wandb/run-*-%s/config.yaml' % (path, ID))[0]
    with open(config_f, 'r') as f:
        conf = yaml.safe_load(f)
    conf = {k: v['value'] for k,v in conf.items() if 'wandb' not in k}
    conf['normed'] = True
    conf['folded'] = True
    aux = re.findall('\/run-(\d+\_\d+?)-\S+\/', config_f)
    conf['date']   = aux[0] if len(aux) != 0 else ''
    conf['ID'] = ID
    
    print('Loading from... \n', fname)
    
    if conf['architecture'] == 'tcn':
        vae = VAE_TCN(latent_dim  = conf['latent_dim'],
                      seq_len     = conf['sequence_lenght'], 
                      kernel_size = conf['kernel_size'], 
                      hidden_dim  = conf['hidden_size'], 
                      nlevels     = conf['num_layers'], 
                      n_feats     = conf['n_feats'], 
                      dropout     = conf['dropout'], 
                      return_norm = conf['normed'], 
                      latent_mode = conf['latent_mode'], 
                      lab_dim     = conf['label_dim'], 
                      phy_dim     = conf['physics_dim'],
                      feed_pp     = True if conf['feed_pp'] == 'T' else False)
    elif conf['architecture'] in['lstm', 'gru']:
        vae = VAE_RNN(latent_dim  = conf['latent_dim'], 
                      seq_len     = conf['sequence_lenght'], 
                      hidden_dim  = conf['hidden_size'], 
                      n_layers    = conf['num_layers'],
                      rnn         = conf['architecture'], 
                      n_feats     = conf['n_feats'], 
                      dropout     = conf['dropout'], 
                      return_norm = conf['normed'], 
                      latent_mode = conf['latent_mode'],
                      lab_dim     = conf['label_dim'], 
                      phy_dim     = conf['physics_dim'])
    state_dict = torch.load(fname, map_location=device)
    if list(state_dict.keys())[0].split('.')[0] == 'module':
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
    else:
        new_state_dict = state_dict
    vae.load_state_dict(new_state_dict)
    vae.eval()
    vae.to(device)
    print('Is model in cuda? ', next(vae.parameters()).is_cuda)
    
    return vae, conf


def evaluate_encoder(model, dataloader, params, 
                     n_classes=5, force=False, device='cpu'):
    """Creates a joint plot of features, used during training, figures
    are W&B ready

    Parameters
    ----------
    model      : pytorch obejct
        model to be evaluated
    dataloader : pytorch object
        dataloader object with data to be evaluated with model
    params     : dictionary
        dictionary of model configuration parameters
    n_classes  : int
        number of unique classes/labels availables in the data
    force      : bool, optional
        wheather to force model evaluation or load values from file archive
    device     : str, optional
        device where model runs, gpu or cpu

    Returns
    -------
    mu_df
        Pandas dataframe of mu values, last column are the labels 
    std_df
        Pandas dataframe of std values, last column are the labels 
    """
    
    fname_mu = '%s/wandb/run-%s-%s/latent_space_mu.txt' % (path, 
params['date'], params['ID'])
    fname_std = '%s/wandb/run-%s-%s/latent_space_std.txt' % (path, params['date'], params['ID'])
    fname_lbs = '%s/wandb/run-%s-%s/labels.txt' % (path, params['date'], params['ID'])
    if os.path.exists(fname_mu) & os.path.exists(fname_std) & os.path.exists(fname_lbs) &  ~force:
        print('Loading from files...')
        mu = np.loadtxt(fname_mu)
        std = np.loadtxt(fname_std)
        labels = np.loadtxt(fname_lbs, dtype=np.str)
    
    else:
        print('Evaluating Encoder...')
        time_start = datetime.datetime.now()
        
        mu, logvar, xhat, labels = [], [], [], []
        with tqdm_notebook(total=len(dataloader)) as pbar:
            for i, (data, label, onehot, pp) in enumerate(dataloader):
                data = data.to(device)
                onehot = onehot.to(device)
                pp = pp.to(device)
                cc = torch.cat([onehot, pp], dim=1)
                if params['label_dim'] > 0 and params['physics_dim'] > 0:
                    mu_, logvar_ = model.encoder(data, label=onehot, phy=pp)
                elif params['label_dim'] > 0 and params['physics_dim'] == 0:
                    mu_, logvar_ = model.encoder(data, label=onehot)
                elif params['label_dim'] == 0:
                    mu_, logvar_ = model.encoder(data)
                else:
                    print('Check conditional dimension...')
                mu.extend(mu_.data.cpu().numpy())
                logvar.extend(logvar_.data.cpu().numpy())
                labels.extend(label)
                torch.cuda.empty_cache()
                pbar.update()
        mu = np.array(mu)
        std = np.exp(0.5 * np.array(logvar))

        np.savetxt(fname_mu, mu)
        np.savetxt(fname_std, std)
        np.savetxt(fname_lbs, np.asarray(labels), fmt='%s')
        elap_time = datetime.datetime.now() - time_start
        print('Elapsed time  : %.2f s' % (elap_time.seconds))
        print('##'*20)
        
    mu_df = pd.DataFrame(mu)
    std_df = pd.DataFrame(std)
        
    mu_df['class'] = labels
    std_df['class'] = labels
    
    return mu_df, std_df

    
def plot_wall_lcs(lc_gen, lc_real, cls=[], lc_gen2=None, save=False):
    """Creates a wall of light curves plot with real and reconstruction
    sequences, paper-ready.

    Parameters
    ----------
    lc_gen  : numpy array
        light curves generated by the VAE model
    lc_real : numpy array
        real light curves overlayed in the plot
    cls     : list, optional
        list with corresponding lables to be displayed as legends
    lc_gen2 : numpy array, optional
        array with second set of generated light curves if desired
    save    : bool, optional
        wheather to save or not the figure
        
    Returns
    -------
        display figure
    """
    
    if len(cls) == 0:
        cls = [''] * len(lc_gen)
    plt.close()
    fig, axis = plt.subplots(nrows=8, ncols=3, 
                             figsize=(16,14),
                             sharex=True, sharey=True)
    
    for i, ax in enumerate(axis.flat):
        ax.errorbar(lc_real[i, :, 0],
                    lc_real[i, :, 1],
                    yerr=lc_real[i, :, 2],
                    fmt='.', c='gray', alpha=.5)

        ax.errorbar(lc_gen[i, :, 0],
                    lc_gen[i, :, 1], 
                    yerr=None,
                    fmt='.', c='royalblue', label=cls[i])
        if lc_gen2 is not None:
            ax.errorbar(lc_gen2[i, :, 0],
                        lc_gen2[i, :, 1], 
                        yerr=None,
                        fmt='.', c='g', alpha=.7)
        if cls[0] != '':
            ax.legend(loc='lower left')
    
    axis[-1,1].set_xlabel('Phase', fontsize=20)
    axis[4,0].set_ylabel('Normalized Magnitude', fontsize=20)
    #mytitle = fig.suptitle('', fontsize=20, y=1.05)

    fig.subplots_adjust(hspace=0, wspace=0)
    axis[0,0].invert_yaxis()
    #for i, ax in enumerate(axis.flat):
    #    ax.invert_yaxis()
    #plt.tight_layout()
    if save:
        plt.savefig('%s/paper_figures/recon_lc_examples_%s.pdf' % 
                    (path, ID), format='pdf', bbox_inches='tight')
    plt.show()
    return 



def scatter_hue(x, y, labels, disc=True, c_label=''):
    """Creates a wall of light curves plot with real and reconstruction
    sequences, paper-ready.

    Parameters
    ----------
    x      : array
        data to be plotted in horizontal axis
    y      : array
        data to be plotted in vertical axis
    labels : list, optional
        list with corresponding lables to be displayed as legends
    disc : bool, optional
        wheather the axis used for coloring is discrete or not
    c_label    : bool, optional
        name of color dimension
        
    Returns
    -------
        display figure
    """
    
    fig = plt.figure(figsize=(12,9))
    if disc:
        c = cm.Dark2_r(np.linspace(0,1,len(set(labels))))
        for i, cls in enumerate(set(labels)):
            idx = np.where(labels == cls)[0]
            plt.scatter(x[idx], y[idx], marker='.', s=20,
                        color=c[i], alpha=.7, label=cls)
    else:
        plt.scatter(x, y, marker='.', s=20,
                    c=labels, cmap='coolwarm_r', alpha=.7)
        plt.colorbar(label=c_label)
        
    plt.xlabel('embedding 1')
    plt.ylabel('embedding 2')
    plt.legend(loc='best', fontsize='x-large')
    plt.show()