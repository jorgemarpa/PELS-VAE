import sys
import numpy as np
import pandas as pd
import torch
import gzip
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn import preprocessing
from src.utils import normalize_each, normalize_glob, return_dt

local_root = '/Users/jorgetil/Google Drive/Colab_Notebooks/data'
colab_root = '/content/drive/My Drive/Colab_Notebooks/data'
exalearn_root = '/home/jorgemarpa/data'

## load pkl synthetic light-curve files to numpy array
class Astro_lightcurves(Dataset):
    """
    Dataset class that loads OGLE light curves and its corresponding 
    metadata. This class provides dataloader, does data normalization 
    and scaling (if requested)

    ...

    Attributes
    ----------
    lcs              : array
        array with OGLE light curves
    meta             : pandas dataframe
        data frame with corresponding metadata
    labels           : array
        array with corresponding labels
    label_int_enc    : sklearn encoder
        scikit-learn integer encoder for string label values
    labels_int       : array
        array of integer encoded labels
    label_onehot_enc : sklearn encoder
        scikit-learn one-hot encoder for string label vales
    labels_onehot    : array
        one-hot encoding of label values
    phy_names        : list
        list of stellar parameters names to be provided by the dataloader
    mm_scaler        : sklearn scaler
        scikit-learn min-max scaler for stellar parameters
    meta_p : array
        min-max-scaled physical parameters

    Methods
    -------
    __getitem__(self, index)
        return data in the index position
    __len__(self)
        return the total length of the entire dataset
    drop_class(self, name)
        drop an specific class determined by "name"
    only_class(self, name)
        drop all labels, but "name"
    remove_nan(self)
        remove nan values according to the list of physical parameters used
    class_value_counts(self)
        print the value counts per label
    get_dataloader(self, batch_size=32, shuffle=True,
                   test_split=0.2, random_seed=42)
        return a dataloader object
    """
    def __init__(self, survey='OGLE3', band='I',
                 use_time=True, use_err=True,
                 norm=True, folded=True, machine='Jorges-MBP',
                 seq_len=600, phy_params='',
                 subsample=False):
        """
        Parameters
        ----------
        survey     : str
            Name of survey to be used (only OGLE3 available for now)
        band       : str
            Name of passband for a given survey name 
            (OGLE3 uses I-band light curves for now)
        use_time   : bool, optional
            return light curves with time or not
        use_err    : bool, optional
            return light curves with error measurements or not
        norm       : bool, optional
            normalize light curves or not
        folded     : bool, optional
            use folded light curves or not
        machine    : bool, optional
            which machine is been used (colab, exalearn, local)
        seq_len    : bool, optional
            length of the light curves to be used
        phy_params : bool, optional
            which physical parameters will be provided with the loader
        subsample  : bool, optional
            wheather to subsample the entire dataset
        """
        
        if machine == 'Jorges-MBP':
            root = local_root
        elif machine == 'colab':
            root = colab_root
        elif machine == 'exalearn':
            root = exalearn_root
        else:
            print('Wrong machine, please select loca, colab or exalearn')
            sys.exit()
        if not folded:
            data_path = ('%s/time_series/real' % (root) +
                        '/%s_lcs_%s_meta_snr5_augmented_trim%i.pkl'
                        % (survey, band, seq_len))
        else:
            data_path = ('%s/time_series/real' % (root) +
                        '/%s_lcs_%s_meta_snr5_augmented_folded_trim%i.npy.gz'
                        % (survey, band, seq_len))
        print('Loading from:\n', data_path)
        with gzip.open(data_path, 'rb') as f:
            self.aux = np.load(f, allow_pickle=True)
        self.lcs = self.aux.item()['lcs']
        self.meta = self.aux.item()['meta']
        del self.aux
        if subsample:
            idx = np.random.randint(0, self.lcs.shape[0], 20000)
            self.lcs = self.lcs[idx]
            self.meta = self.meta.iloc[idx].reset_index(drop=True)
        self.labels = self.meta['Type'].values
        ## integer encoding of labels
        self.label_int_enc = preprocessing.LabelEncoder()
        self.label_int_enc.fit(self.labels)
        self.labels_int = self.label_int_enc.transform(self.labels)
        ## one-hot encoding of labels
        self.label_onehot_enc = preprocessing.OneHotEncoder(sparse=False,
                                                            categories='auto',
                                                            dtype=np.float32)
        self.label_onehot_enc.fit(self.labels.reshape(-1, 1))
        self.labels_onehot = self.label_onehot_enc.transform(self.labels.reshape(-1, 1))

        if use_time and not use_err:
            self.lcs = self.lcs[:, :, 0:2]
        if not use_time and not use_err:
            self.lcs = self.lcs[:, :, 1:2]

        if not 'folded' in data_path:
            self.lcs = return_dt(self.lcs)
        if norm:
            self.lcs = normalize_each(self.lcs, 
                                      n_feat=self.lcs.shape[2],
                                      scale_to=[.0001, .9999],
                                      norm_time=use_time)
        
        self.phy_names = []
        if len(phy_params) > 0:
            if 'p' in phy_params or 'P' in phy_params:
                self.phy_names.append('Period')
            if 't' in phy_params or 'T' in phy_params:
                self.phy_names.append('teff_val')
            if 'm' in phy_params or 'M' in phy_params:
                self.phy_names.append('[Fe/H]_J95')
            if 'c' in phy_params or 'C' in phy_params:
                self.phy_names.append('bp_rp')
            if 'a' in phy_params or 'A' in phy_params:
                self.phy_names.append('phot_g_mean_abs_mag')
            if 'r' in phy_params or 'R' in phy_params:
                self.phy_names.append('radius_val')
            if 'l' in phy_params or 'L' in phy_params:
                self.phy_names.append('lum_val')
            self.phy_aux = self.phy_names
        else:
            self.phy_aux = ['Period']
            
        self.mm_scaler = preprocessing.MinMaxScaler()
        self.mm_scaler.fit(self.meta.loc[:, self.phy_aux].values.astype(np.float32))
        self.meta_p = self.mm_scaler.transform(
            self.meta.loc[:, self.phy_aux].values.astype(np.float32))


    def __getitem__(self, index):
        """Return the item in the position index

        Parameters
        ----------
        index : int
            index position to be returned
        
        Returns
        -------
        lc
            light curve
        label
            corresponding label
        onehot
            corresponding one-hot encoding of the label
        meta_p
            corresponding scaled physical parameter values
        """
        lc = self.lcs[index]
        label = self.labels[index]
        meta = self.meta.iloc[index]
        onehot = self.labels_onehot[index]
        meta_p = self.meta_p[index]
        return lc, label, onehot, meta_p


    def __len__(self):
        return len(self.lcs)


    def drop_class(self, name):
        """Remove all labels that matche the string "name"

        Parameters
        ----------
        name : str, optional
            label name to be dropped from the dataset
        """
        idx = np.where(self.labels != name)[0]
        self.lcs = self.lcs[idx]
        self.labels = self.labels[idx]
        self.meta = self.meta.iloc[idx].reset_index(drop=True)
        self.meta_p = self.meta_p[idx]
        self.labels_onehot = self.labels_onehot[idx]
        self.labels_int = self.labels_int[idx]


    def only_class(self, name):
        """Only keep items with labels that match "name"

        Parameters
        ----------
        name : str, optional
            label name to be keep from the dataset
        """
        idx = np.where(self.labels == name)[0]
        self.lcs = self.lcs[idx]
        self.labels = self.labels[idx]
        self.meta = self.meta.iloc[idx].reset_index(drop=True)
        self.meta_p = self.meta_p[idx]
        self.labels_onehot = self.labels_onehot[idx]
        self.labels_int = self.labels_int[idx]


    def remove_nan(self):
        """Remove all items with nan values in their correspoding physical
        parameters

        Parameters
        ----------
        """
        idx = self.meta.dropna(axis='index',
                               subset=self.phy_aux).index.values
        self.lcs = self.lcs[idx]
        self.labels = self.labels[idx]
        self.meta = self.meta.iloc[idx]
        self.meta_p = self.meta_p[idx]
        self.labels_onehot = self.labels_onehot[idx]
        self.labels_int = self.labels_int[idx]


    def class_value_counts(self):
        """Print value counts of labels

        Parameters
        ----------
        """
        print(self.meta.Type.value_counts())


    def get_dataloader(self, batch_size=32, shuffle=True,
                       test_split=0.2, random_seed=42):
        """Creates a data loader object to be used during model training

        Parameters
        ----------
        batch_size : int
            sixe of the batch
        shuffle    : bool, optional
            wheather to shuffle the data
        test_split : float, optional
            fraction of the dataset used for testing
        random_seed : int, optional
            random seed used for shufling data and train/test split
        
        Returns
        -------
        train_loader
            data loader used for trianing
        test_loader
            data loader used for testing/validation
        """

        np.random.seed(random_seed)
        if test_split == 0.:
            train_loader = DataLoader(self, batch_size=batch_size,
                                      shuffle=shuffle, drop_last=False)
            test_loader = None
        else:
            # Creating data indices for training and test splits:
            dataset_size = len(self)
            indices = list(range(dataset_size))
            split = int(np.floor(test_split * dataset_size))
            np.random.shuffle(indices)
            train_indices, test_indices = indices[split:], indices[:split]

            # Creating PT data samplers and loaders:
            train_sampler = SubsetRandomSampler(train_indices)
            test_sampler = SubsetRandomSampler(test_indices)

            train_loader = DataLoader(self, batch_size=batch_size,
                                      sampler=train_sampler, drop_last=False)
            test_loader = DataLoader(self, batch_size=batch_size,
                                      sampler=test_sampler, drop_last=False)

        return train_loader, test_loader
