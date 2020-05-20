"""VAE main training script

This script allows the user to train a VAE model using OGLE data loaded
with the 'dataset.py' class, VAE model located in 'vae_model.py' class, and
the trainig loop coded in 'vae_training.py'. THe script also uses Weight & Biases
framework to log metrics, model hyperparameters, configuration parameters, and
training figures.

This file contains the following
functions:

    * run_code - runs the main code
"""

import sys, os
import datetime
import argparse
import torch
import torch.optim as optim
import numpy as np
from src.datasets import *
from src.utils import *
from src.vae_models import *
from src.vae_training import Trainer

import wandb

rnd_seed = 13
np.random.seed(rnd_seed)
torch.manual_seed(rnd_seed)
torch.cuda.manual_seed_all(rnd_seed)
#torch.backends.cudnn.deterministic = True
## allows cudnn to look for the optimal algorithm config 
## for faster runtime, only useful when network inputs 
## are always same size
torch.backends.cudnn.benchmark = True    
#random.seed(rnd_seed)
#os.environ['PYTHONHASHSEED'] = str(rnd_seed)

## Config ##
parser = argparse.ArgumentParser(description='Variational Auto Encoder (VAE)'+
                                 'to produce synthetic astronomical time series')
parser.add_argument('--dry-run', dest='dry_run', action='store_true',
                    default=False,
                    help='Load data and initialize models [False]')
parser.add_argument('--machine', dest='machine', type=str, default='Jorges-MBP',
                    help='were to is running (Jorges-MBP, colab, [exalearn])')

parser.add_argument('--data', dest='data', type=str, default='OGLE3',
                    help='data used for training ([OGLE3], EROS2)')
parser.add_argument('--use-err', dest='use_err', type=str, default='T',
                    help='use magnitude errors ([T],F)')
parser.add_argument('--cls', dest='cls', type=str, default='all',
                    help='drop or select ony one class '+
                    '([all],drop_"vartype",only_"vartype")')

parser.add_argument('--lr', dest='lr', type=float, default=1e-4,
                    help='learning rate [1e-4]')
parser.add_argument('--lr-sch', dest='lr_sch', type=str, default=None,
                    help='learning rate shceduler '+
                    '([None], step, exp,cosine, plateau)')
parser.add_argument('--beta', dest='beta', type=str, default='1',
                    help='beta factor for latent KL div ([1],step)')
parser.add_argument('--batch-size', dest='batch_size', type=int, default=128,
                    help='batch size [128]')
parser.add_argument('--num-epochs', dest='num_epochs', type=int, default=150,
                    help='total number of training epochs [150]')

parser.add_argument('--cond', dest='cond', type=str, default='T',
                    help='label conditional VAE (F,[T])')
parser.add_argument('--phy', dest='phy', type=str, default='',
                    help='physical parameters to use for conditioning ([],[tm])')
parser.add_argument('--latent-dim', dest='latent_dim', type=int, default=6,
                    help='dimension of latent space [6]')
parser.add_argument('--latent-mode', dest='latent_mode', type=str,
                    default='repeat',
                    help='wheather to sample from a 3d or 2d tensor '+
                    '([repeat],linear,convt)')
parser.add_argument('--arch', dest='arch', type=str, default='tcn',
                    help='architecture for Enc & Dec ([tcn],lstm,gru)')
parser.add_argument('--transpose', dest='transpose', type=str, default='F',
                    help='use tranpose convolution in Dec ([F],T)')
parser.add_argument('--units', dest='units', type=int, default=32,
                    help='number of hidden units [32]')
parser.add_argument('--layers', dest='layers', type=int, default=5,
                    help='number of layers/levels for lstm/tcn [5]')
parser.add_argument('--dropout', dest='dropout', type=float, default=0.2,
                    help='dropout for lstm/tcn layers [0.2]')
parser.add_argument('--kernel-size', dest='kernel_size', type=int, default=5,
                    help='kernel size for tcn conv, use odd ints [5]')

parser.add_argument('--comment', dest='comment', type=str, default='',
                    help='extra comments')
args = parser.parse_args()

machine = args.machine

## data params
data = args.data
if data == 'EROS2':
    seq_len = 150
    band = 'B'
elif data == 'OGLE3':
    seq_len = 600
    band = 'I'
else:
    seq_len = args.seq_len
use_time = True
use_err = True
shuffle = True
norm = True
folded = True
cls = args.cls

cmt = args.comment

## training params
lr = args.lr
lr_sch = args.lr_sch
batch_size = args.batch_size
num_epochs = args.num_epochs
beta = args.beta

## architecture params
cond_l = True if args.cond == 'T' else False
phy = args.phy
cond_p = True if len(phy) > 0 else False

latent_dim = args.latent_dim
latent_mode = args.latent_mode
n_units = args.units
nlevels = args.layers
dropout = args.dropout
arch = args.arch      # lstm, gru, tcn
transpose = True if args.transpose == 'T' else False
kernel_size = args.kernel_size

if use_time and use_err:
    n_feat = 3
elif use_time and not use_err:
    n_feat = 2
elif not use_time and use_err:
    n_feat = 2
elif not use_time and not use_err:
    n_feat = 1


## Model names and log directory
if cond_l or cond_p:
    model_name = 'cVAE'
else:
    model_name = 'VAE'

# Initialize W&B project
wandb.init(project="Phy-VAE", notes=cmt, tags=[data, cmt, phy])

# Define hyper-parameters
config = wandb.config
config.architecture = arch
config.hidden_size = n_units
config.num_layers = nlevels
config.kernel_size = kernel_size
config.dropout = dropout
config.latent_dim = latent_dim
config.latent_mode = latent_mode
config.transpose = transpose

config.epochs = num_epochs
config.learning_rate = lr
config.learning_rate_scheduler = lr_sch
config.batch_size = batch_size
config.beta_vae = beta

config.data = data
config.classes = cls
config.sequence_lenght = seq_len
config.n_feats = n_feat
config.phys_params = phy.upper()


## run main program
def run_code():

    ## Load Data ##
    if data in ['EROS2', 'OGLE3']:
        dataset = Astro_lightcurves(survey=data,
                                    band=band,
                                    use_time=use_time,
                                    use_err=use_err,
                                    norm=norm,
                                    folded=folded,
                                    machine=machine,
                                    seq_len=seq_len,
                                    phy_params=phy,
                                    subsample=False)

        if cls.split('_')[0] == 'drop':
            dataset.drop_class(cls.split('_')[1])
        elif cls.split('_')[0] == 'only':
            dataset.only_class(cls.split('_')[1])

        print('Using physical parameters: ', dataset.phy_names)
        dataset.remove_nan()

        dataset.class_value_counts()
    elif data == 'synt':
        dataset = Synt_lightcurves(use_time=use_time,
                                   use_err=use_err,
                                   norm=norm,
                                   folded=folded,
                                   colab=colab,
                                   seq_len=seq_len)

    else:
        print('Error: Wrong dataset (eros, synt)...')
        raise

    if len(dataset) == 0:
        print('No items in training set...')
        print('Exiting!')
        sys.exit()

    ## show real light curves wall if running in Colab
    rnd_idx = np.random.randint(0, len(dataset), 8)
    lc_ex, labels_ex = dataset.lcs[rnd_idx], dataset.labels[rnd_idx]

    ## data loaders for training and testing
    train_loader, test_loader = dataset.get_dataloader(batch_size=batch_size,
                                                       shuffle=True,
                                                       test_split=.2,
                                                       random_seed=rnd_seed)
    print('\nTraining lenght: ', len(train_loader) * batch_size)
    print('Test lenght    : ', len(test_loader) * batch_size)

    config.label_dim = dataset.labels_onehot[0].shape[0] if cond_l else 0
    config.physics_dim = len(dataset.phy_names)
    config.feed_pp = 'F'
    print('Label dimension : ', config.label_dim)
    print('Physic dimension: ', config.physics_dim)

    ## Define GAN model, Ops, and Train ##
    if arch == 'tcn':
        vae = VAE_TCN(latent_dim, seq_len, kernel_size, n_units, nlevels,
                     n_feats=n_feat, dropout=dropout, return_norm=norm,
                     latent_mode=latent_mode, transpose=transpose,
                     lab_dim=config.label_dim, 
                     phy_dim=config.physics_dim,
                     feed_pp=True if config.feed_pp == 'T' else False)
    elif arch in ['lstm', 'gru']:
        vae = VAE_RNN(latent_dim, seq_len, n_units, nlevels,
                      rnn=arch, n_feats=n_feat, dropout=dropout, 
                      return_norm=norm, latent_mode=latent_mode,
                      lab_dim=config.label_dim, 
                      phy_dim=config.physics_dim)
    else:
        print('Error: wrong architecture, select between (tcn, lstm) ')
        raise

    wandb.watch(vae, log='gradients')

    print('Summary:')

    config.n_train_params = count_parameters(vae)
    print('Num of trainable params: ', count_parameters(vae))
    print(vae)
    print('\n')

    # Initialize optimizers
    optimizer = optim.Adam(vae.parameters(), lr=lr)

    # Learning Rate scheduler
    if lr_sch == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=20,
                                              gamma=0.5)
    elif lr_sch == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                     gamma=0.985)
    elif lr_sch == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         T_max=50,
                                                         eta_min=1e-5)
    elif lr_sch == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=.5,
                                                         verbose=True)
    else:
        scheduler = None

    print('Optimizer    :', optimizer)
    print('LR Scheduler :', scheduler.__class__.__name__)

    # Train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print('########################################')
    print('########  Running in %4s  #########' % (device))
    print('########################################')

    trainer = Trainer(vae, optimizer, batch_size, wandb,
                      print_every=200,
                      device=device,
                      scheduler=scheduler, cond_l=cond_l,
                      cond_p=cond_p, beta=beta)

    if args.dry_run:
        print('******** DRY RUN ******** ')
        return

    trainer.train(train_loader, test_loader, num_epochs, (lc_ex, labels_ex),
                  machine=machine, save=True, early_stop=False)


if __name__ == "__main__":
    print('Running in: ', machine, '\n')
    for key, value in vars(args).items():
        print('%15s\t: %s' % (key, value))
    print('\nModel name\n', model_name)

    run_code()
