# Astronomical Time Series

Light curves taken from [OGLE 3](http://www.astrouw.edu.pl/ogle/ogle3/OIII-CVS/), which contains the following variability classes: 
* Eclipsing Binaries
* Anomalous Cepheids
* Cepheids
* Type II Cepheids
* RR Lyrae
* Long Period Variables
* Ellipsoidal Variables
* Delta Scuti

Training data is avalaible [here](https://zenodo.org/record/3820679#.XsW12RMzaRc).

#### Samples
![Light Curve samples](https://github.com/jorgemarpa/PELS-VAE/paper_figures/OGLE3_lcs_ex.pdf)

#### Gaia DR2 parameters
![Joint distribution](https://github.com/jorgemarpa/PELS-VAE/paper_figures/phys_params_joint.pdf)

## Usage

Use `vae_main.py` to train a cVAE model with the following parameters:

```
  --dry-run             Only load data and initialize model [False]
  --machine             were to is running ([Jorges-MBP], colab, exalearn)
  --data                data used for training (OGLE3)
  --use-err             use magnitude errors ([T],F)
  --cls                 drop or select ony one class
                        ([all],drop_"vartype",only_"vartype")
  --lr                  learning rate [1e-4]
  --lr-sch              learning rate shceduler ([None], step, exp,cosine,
                        plateau)
  --beta                beta factor for latent KL div ([1],step)
  --batch-size          batch size [128]
  --num-epochs          total number of training epochs [150]
  --cond                label conditional VAE (F,[T])
  --phy                 physical parameters to use for conditioning ([],[tm])
  --latent-dim          dimension of latent space [6]
  --latent-mode         wheather to sample from a 3d or 2d tensor
                        ([repeat],linear,convt)
  --arch                architecture for Enc & Dec ([tcn],lstm,gru)
  --transpose           use tranpose convolution in Dec ([F],T)
  --units               number of hidden units [32]
  --layers              number of layers/levels for lstm/tcn [5]
  --dropout             dropout for lstm/tcn layers [0.2]
  --kernel-size         kernel size for tcn conv, use odd ints [5]
  --comment             extra comments
```

Architecture available are [TCN, LSTM, GRU]. The encoder-decoder contain a sequential artchitecture followed by a set of dense layers.

This train the models and generate a tensorboard event log (located in ./logs) of the training progress.

#### Recontruction examples
![Light Curve reconstruction](https://github.com/jorgemarpa/PELS-VAE/paper_figures/recon_lc_examples_YES.png)

## Sources and inspiration

* https://www.jeremyjordan.me/variational-autoencoders/
* https://github.com/wiseodd/generative-models
* https://github.com/kefirski/pytorch_RVAE
