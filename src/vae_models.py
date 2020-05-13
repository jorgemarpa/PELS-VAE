import torch
import torch.nn as nn
import torch.nn.functional as F
from src.tcn import TemporalConvNet


class VAE_TCN(nn.Module):
    """
    A class that creates a Variational Autoencoder
    using Temporal Conv Net as sequential architecture.

    ...

    Attributes
    ----------
    num_channels_enc : list
        list with number of channels per temporal block
    latent_expand    : pytorch sequential
        layer that expand the latent code to match decoder's TCN input
    enc_tcn          : pytorch module
        Encoders TCN layer 
    enc_linear       : pytorch sequential
        Block of linear + activation func + dropout layers after TCN layer
    enc_linear_mu    : pytorch module
        Linear layer that outputs mean values of Gaussian dist
    enc_linear_std   : pytorch module
        Linear layer that outputs logvar values of Gaussian dist
    dec_tcn          : pytorch module
        Decoders TCN layer
    dec_linear       : pytorch sequential
        Decoders dense layer

    Methods
    -------
    encoder(self, x, label=None, phy=None)
        Encoder module
    decoder(self, z, dt, label=None, phy=Nne)
        Decoder module
    reparameterize(self, mu, logvar)
        Reparametrization used to sample from N(mu,logvar)
    forward(self, x, label=None, phy=None)
        VAE forward pass
    """
    
    def __init__(self, latent_dim, seq_len, kernel_size,
                 hidden_dim, nlevels,
                 n_feats=3, dropout=0., return_norm=True,
                 transpose=False, latent_mode='repeat',
                 lab_dim=0, phy_dim=0, feed_pp=True):
        """
        Parameters
        ----------
        latent_dim  : int
            dimension of the latent space
        seq_len     : int
            length of sequences
        kernel_size : int
            size of convolution kernel
        hidden_dim  : int
            size of TCN hidden dimension
        nlevels     : int
            number of temporal blocks
        n_feats     : int
            number of used features (time, mag, mag_err)
        dropout     : float
            dropout probability
        return_norm : bool
            return values normalized or not
        transpose   : bool
            use transpose convolutions or not
        latent_mode : str
            type of latent expansion mode [linear, convt, repeat]
        lab_dim     : int
            number of total unique labels used
        phy_dim     : int
            number of stellar parameters used
        feed_pp     : bool
            wheather to feed physical parameters to decoder or not
        """
        super(VAE_TCN, self).__init__()

        self.latent_dim = latent_dim
        self.phy_dim = phy_dim
        self.lab_dim = lab_dim
        self.return_norm = return_norm
        self.seq_len = seq_len
        self.feed_pp = feed_pp
        ## all layers have same number of hidden units,
        ## different units per layers are possible by specifying them
        ## as a list, e.g. [32,16,8]
        self.num_channels_enc = [hidden_dim] * nlevels
        self.num_channels_enc[0] = 8
        self.num_channels_enc[1] = 16
        self.num_channels_enc[2] = 32
        self.num_channels_dec = [hidden_dim] * nlevels
        self.num_channels_dec[-1] = 8
        self.num_channels_dec[-2] = 16
        self.num_channels_dec[-3] = 32
        self.latent_mode = latent_mode

        if latent_mode == 'linear':
            self.latent_expand = nn.Sequential(
                nn.Linear(latent_dim, latent_dim * seq_len),
                nn.ReLU())
        elif latent_mode == 'convt':
            self.latent_expand = nn.Sequential(
                nn.ConvTranspose1d(latent_dim, latent_dim,
                                   kernel_size=seq_len, dilation=1),
                nn.ReLU())

        self.enc_tcn = TemporalConvNet(n_feats,
                                       self.num_channels_enc,
                                       kernel_size,
                                       dropout=dropout)
        self.enc_linear = nn.Sequential(
            nn.Linear(hidden_dim + lab_dim + phy_dim,
                      16),
            nn.ReLU(),
            nn.Dropout(dropout))
            #nn.Linear(32, 16),
            #nn.ReLU(),
            #nn.Dropout(dropout))

        self.enc_linear_mu = nn.Linear(16, latent_dim)
        self.enc_linear_std = nn.Linear(16, latent_dim)

        if not transpose:
            self.dec_tcn = TemporalConvNet(1 + latent_dim + lab_dim + (phy_dim if feed_pp else 0),
                                           self.num_channels_dec,
                                           kernel_size,
                                           dropout=dropout)
        else:
            self.dec_tcn = TemporalConvNet(1 + latent_dim + lab_dim + (phy_dim if feed_pp else 0),
                                           self.num_channels_dec,
                                           kernel_size,
                                           dropout=dropout,
                                           transpose=True)
        self.dec_linear = nn.Linear(self.num_channels_dec[-1], 1)


    def encoder(self, x, label=None, phy=None):
        """Encoder module, inputs light curves, extract features 
        using a TCN layer, append metadata to the extractedfeature,
        then used a sequence of dense layers to predict the 
        distribution parameters of the latent space as a Gaussian 
        distribution

        Parameters
        ----------
        x     : tensor
            tensor of light curves [bacth, time stamps, features]
        label : tensor, optional
            tensor of one-hot encoded labels
        phy   : tensor, optional
            tensor of physical parameters
        
        Returns
        -------
        mu
            mean value of predicted normal distirbution
        logvar
            log of variance value of predicted normal distirbution
        """
        eh1 = self.enc_tcn(x.transpose(1, 2)).transpose(1, 2)
        #eh1 = eh1.flatten(start_dim=1)
        eh1 = eh1[:,-1,:]
        
        if self.lab_dim > 0 and label is not None:
            eh1 = torch.cat([eh1, label], dim=1)
        if self.phy_dim > 0 and phy is not None:
            eh1 = torch.cat([eh1, phy], dim=1)
            
        eh2 = self.enc_linear(eh1)
        mu = self.enc_linear_mu(eh2)
        logvar = self.enc_linear_std(eh2)
        return mu, logvar


    def decoder(self, z, dt, label=None, phy=None):
        """Decoder module, inputs laten code and one-hot encoded
        labels (physical params can be added too). Latent code
        and labels are concatenated an tagged with timestamp values (dt).
        Then, the resulting vector is expanded to match the sequence length,
        there are three methods to do this, using a linear layer, transpose
        convolutions, or repeat vetor (recommended).

        Parameters
        ----------
        z     : tensor
            latent code
        dt    : tensor
            tensor of delta times (time stamps)
        label : tensor, optional
            tensor of one-hot encoded labels
        phy   : tensor, optional
            tensor of physical parameters
        
        Returns
        -------
        xhat
            reconstructed light curve
        """
        if self.lab_dim > 0 and label is not None:
            z = torch.cat([z, label], dim=1)
        if self.phy_dim > 0 and phy is not None and self.feed_pp:
            z = torch.cat([z, phy], dim=1)
            
        if self.latent_mode == 'linear':
            z_ = self.latent_expand(z).unsqueeze(-1).view(-1,
                                                          self.seq_len,
                                                          self.latent_dim)
        elif self.latent_mode == 'convt':
            z_ = self.latent_expand(z.unsqueeze(-1)).transpose(1, 2)
        elif self.latent_mode == 'repeat':
            z_ = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        z_ = torch.cat([dt.unsqueeze(-1), z_], dim=-1)
        dh1 = self.dec_tcn(z_.transpose(1, 2)).transpose(1, 2)
        xhat = self.dec_linear(dh1)
        if self.return_norm:
            xhat = torch.sigmoid(xhat)
        return xhat


    def reparameterize(self, mu, logvar):
        """Reparameterization trick used to allow backpropagation 
        through stochastic process

        Parameters
        ----------
        mu     : tensor
            tensor of mean values
        logvar : tensor
            tensor of log variance values
        
        Returns
        -------
        latent_code
            tensor of sample latent codes
        """
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std, requires_grad=False)
        return mu + eps*std


    def forward(self, x, label=None, phy=None):
        """Forward pass of VAE: endcoder -> latent-code sampling -> decoder.

        Parameters
        ----------
        x     : tensor
            tensor of light curves
        label : tensor, optional
            tensor of one-hot encoded labels
        phy   : tensor, optional
            tensor of physical parameters
        
        Returns
        -------
        xhat
            reconstructed light curve
        mu
            mean values of latent normal distributions
        logvar
            log variance of latent normal distributions
        z
            sampled latent code
        """
        mu, logvar = self.encoder(x, label=label, phy=phy)
        z = self.reparameterize(mu, logvar)
        xhat = self.decoder(z, x[:,:,0], label=label, phy=phy)
        return xhat, mu, logvar, z



class VAE_RNN(nn.Module):
    """
    A class that creates a Variational Autoencoder
    using Recurrent Neural Nets as sequential architecture.

    ...

    Attributes
    ----------
    num_channels_enc : list
        list with number of channels per temporal block
    latent_expand    : pytorch sequential
        layer that expand the latent code to match decoder's TCN input
    enc_lstm         : pytorch module
        Encoders TCN layer 
    enc_linear       : pytorch sequential
        Block of linear + activation func + dropout layers after TCN layer
    enc_linear_mu    : pytorch module
        Linear layer that outputs mean values of Gaussian dist
    enc_linear_std   : pytorch module
        Linear layer that outputs logvar values of Gaussian dist
    dec_lstm         : pytorch module
        Decoders TCN layer
    dec_linear       : pytorch sequential
        Decoders dense layer

    Methods
    -------
    init_weights(self)
        RNN weights and biases initialization
    encoder(self, x, label=None, phy=None)
        Encoder module
    decoder(self, z, dt, label=None, phy=Nne)
        Decoder module
    reparameterize(self, mu, logvar)
        Reparametrization used to sample from N(mu,logvar)
    forward(self, x, label=None, phy=None)
        VAE forward pass
    """
    def __init__(self, latent_dim, seq_len, hidden_dim, n_layers,
                 rnn='lstm', n_feats=3, dropout=0., return_norm=True,
                 latent_mode='repeat', lab_dim=0, phy_dim=0,
                 feed_pp=True):
        """
        Parameters
        ----------
        latent_dim  : int
            dimension of the latent space
        seq_len     : int
            length of sequences
        hidden_dim  : int
            size of RNN hidden dimension
        n_layers    : int
            number of RNNs layers
        rnn         : str
            type of RNN, [LSTM or GRU] are availables
        n_feats     : int
            number of used features (time, mag, mag_err)
        dropout     : float
            dropout probability
        return_norm : bool
            return values normalized or not
        latent_mode : str
            type of latent expansion mode [linear, convt, repeat]
        lab_dim     : int
            number of total unique labels used
        phy_dim     : int
            number of stellar parameters used
        feed_pp     : bool
            wheather to feed physical parameters to decoder or not
        """
        super(VAE_RNN, self).__init__()

        self.latent_dim = latent_dim
        self.return_norm = return_norm
        self.seq_len = seq_len
        self.latent_mode = latent_mode
        self.phy_dim = phy_dim
        self.lab_dim = lab_dim
        self.feed_pp = feed_pp

        if latent_mode == 'linear':
            self.latent_expand = nn.Sequential(
                nn.Linear(latent_dim, latent_dim * seq_len),
                nn.ReLU())
        elif latent_mode == 'convt':
            self.latent_expand = nn.Sequential(
                nn.ConvTranspose1d(latent_dim, latent_dim,
                                   kernel_size=seq_len, dilation=1),
                nn.ReLU())

        ## batch_first --> [batch, seq, feature]
        if rnn == 'lstm':
            self.enc_lstm = nn.LSTM(n_feats, hidden_dim, n_layers,
                                    batch_first=True, dropout=dropout,
                                    bidirectional=False)
            self.dec_lstm = nn.LSTM(1 + latent_dim + lab_dim + (phy_dim if feed_pp else 0),
                                    hidden_dim,
                                    n_layers,
                                    batch_first=True, dropout=dropout,
                                    bidirectional=False)
        elif rnn == 'gru':
            self.enc_lstm = nn.GRU(n_feats, hidden_dim, n_layers,
                                   batch_first=True, dropout=dropout,
                                   bidirectional=False)
            self.dec_lstm = nn.GRU(1 + latent_dim + lab_dim + (phy_dim if feed_pp else 0), 
                                   hidden_dim,
                                   n_layers,
                                   batch_first=True, dropout=dropout,
                                   bidirectional=False)

        self.enc_linear = nn.Sequential(
            nn.Linear(hidden_dim + lab_dim + phy_dim,
                      16),
            nn.ReLU(),
            nn.Dropout(dropout))
            #nn.Linear(32, 16),
            #nn.ReLU(),
            #nn.Dropout(dropout))

        self.enc_linear_mu = nn.Linear(16, latent_dim)
        self.enc_linear_std = nn.Linear(16, latent_dim)


        self.dec_linear = nn.Linear(hidden_dim, 1)

        self.init_weights()


    def init_weights(self):
        """Initialize weight of recurrent layers
        """
        for name, param in self.enc_lstm.named_parameters():
            if 'bias' in name:
                nn.init.normal_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        for name, param in self.dec_lstm.named_parameters():
            if 'bias' in name:
                 nn.init.normal_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)



    def encoder(self, x, label=None, phy=None):
        """Encoder module, inputs light curves, extract features 
        using a TCN layer, append metadata to the extractedfeature,
        then used a sequence of dense layers to predict the 
        distribution parameters of the latent space as a Gaussian 
        distribution

        Parameters
        ----------
        x     : tensor
            tensor of light curves [bacth, time stamps, features]
        label : tensor, optional
            tensor of one-hot encoded labels
        phy   : tensor, optional
            tensor of physical parameters
        
        Returns
        -------
        mu
            mean value of predicted normal distirbution
        logvar
            log of variance value of predicted normal distirbution
        """
        self.enc_lstm.flatten_parameters()
        eh1, h_state = self.enc_lstm(x)
        # eh1 = eh1.flatten(start_dim=1)
        eh1 = eh1[:,-1,:]
        
        if self.lab_dim > 0 and label is not None:
            eh1 = torch.cat([eh1, label], dim=1)
        if self.phy_dim > 0 and phy is not None:
            eh1 = torch.cat([eh1, phy], dim=1)
            
        eh2 = self.enc_linear(eh1)
        mu = self.enc_linear_mu(eh2)
        logvar = self.enc_linear_std(eh2)
        return mu, logvar


    def decoder(self, z, dt, label=None, phy=None):
        """Decoder module, inputs laten code and one-hot encoded
        labels (physical params can be added too). Latent code
        and labels are concatenated an tagged with timestamp values (dt).
        Then, the resulting vector is expanded to match the sequence length,
        there are three methods to do this, using a linear layer, transpose
        convolutions, or repeat vetor (recommended).

        Parameters
        ----------
        z     : tensor
            latent code
        dt    : tensor
            tensor of delta times (time stamps)
        label : tensor, optional
            tensor of one-hot encoded labels
        phy   : tensor, optional
            tensor of physical parameters
        
        Returns
        -------
        xhat
            reconstructed light curve
        """
        if self.lab_dim > 0 and label is not None:
            z = torch.cat([z, label], dim=1)
        if self.phy_dim > 0 and phy is not None and self.feed_pp:
            z = torch.cat([z, phy], dim=1)
            
        if self.latent_mode == 'linear':
            z_ = self.latent_expand(z).unsqueeze(-1).view(-1,
                                                          self.seq_len,
                                                          self.latent_dim)
        elif self.latent_mode == 'convt':
            z_ = self.latent_expand(z.unsqueeze(-1)).transpose(1, 2)
        elif self.latent_mode == 'repeat':
            z_ = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        z_ = torch.cat([dt.unsqueeze(-1), z_], dim=-1)
        self.dec_lstm.flatten_parameters()
        dh1, h_state = self.dec_lstm(z_)
        xhat = self.dec_linear(dh1)
        if self.return_norm:
            xhat = torch.sigmoid(xhat)
        return xhat


    def reparameterize(self, mu, logvar):
        """Reparameterization trick used to allow backpropagation 
        through stochastic process

        Parameters
        ----------
        mu     : tensor
            tensor of mean values
        logvar : tensor
            tensor of log variance values
        
        Returns
        -------
        latent_code
            tensor of sample latent codes
        """
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std, requires_grad=False)
        return mu + eps*std


    def forward(self, x, label=None, phy=None):
        """Forward pass of VAE: endcoder -> latent-code sampling -> decoder.

        Parameters
        ----------
        x     : tensor
            tensor of light curves
        label : tensor, optional
            tensor of one-hot encoded labels
        phy   : tensor, optional
            tensor of physical parameters
        
        Returns
        -------
        xhat
            reconstructed light curve
        mu
            mean values of latent normal distributions
        logvar
            log variance of latent normal distributions
        z
            sampled latent code
        """
        mu, logvar = self.encoder(x, label=label, phy=phy)
        z = self.reparameterize(mu, logvar)
        xhat = self.decoder(z, x[:,:,0], label=label, phy=phy)
        return xhat, mu, logvar, z
