import numpy as np
import datetime
import torch
import torch.nn as nn
from torch.nn import functional as F
from src.utils import *
from src.training_callbacks import EarlyStopping


class Trainer():
    """
    A training class for VAE model. Incorporate batch-training/testing loop, 
    training callbacks, lerning rate scheduler, W&B logger with metrics reports and
    reconstruction plots.

    ...

    Attributes
    ----------
    train_loss  : dict
        dictionary with training metrics
    test_loss   : dict
         dictionary with testing metrics
    num_steps   : int
        number of training steps
    print_every : int
        report metric values every N training steps
    beta        : str
        value of beta hyperparameter that weight the KLD term in the loss function
    wb          : object
        weight and biases logger

    Methods
    -------
    _beta_scheduler(self, epoch, beta0=0., step=50, gamma=0.1)
        beta hyperparameter sheduler
    _loss(self, x, xhat, mu, logvar, train=True, ep=0)
        calculate loss metrics and add them to logger
    _train_epoch(self, data_loader, epoch)
        do model training in a given epoch
    _test_epoch(self, test_loader, epoch)
        do model testing in a given epoch
    _report_train(self, i)
        report training metrics to W&B logger and standard terminal
    _report_test(self, ep)
        report test metrics to W&B logger and standard terminal
    train(self, train_loader, test_loader, epochs, data_ex,
          machine='local', save=True, early_stop=False)
        complete epoch training/validation/report loop
    """
    def __init__(self, model, optimizer, batch_size, wandb,
                 scheduler=None, cond_l=False, cond_p=False,
                 beta='step', print_every=50,
                 device='cpu'):
        """
        Parameters
        ----------
        model       : pytorch module
            pytorch VAE model
        optimizer   : pytorch optimizer
            pytorch optimizer object
        batch_size  : int
            size of batch for training loop
        wandb       : object
            weight and biases logger
        scheduler   : object
            pytorch learning rate schduler, default is None
        cond_l      : bool
            wheather to condition the VAE model with label values or not
        cond_p      : bool
            wheather to condition the VAE model with phyisical parameters or not
        beta        : float/str
            if float, then beta will have always same value. If str == 'step', then
            beta is based on a step scheduler
        print_every : int
            step interval for report printing
        device      : str
            device where model will run (cpu, gpu)
        """
        
        self.device = device
        self.model = model
        if torch.cuda.device_count() > 1 and True:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        print('Is model in cuda? ', next(self.model.parameters()).is_cuda)
        self.opt = optimizer
        self.sch = scheduler
        self.cond_l = cond_l
        self.cond_p = cond_p
        self.batch_size = batch_size
        self.train_loss = {'KL_latent': [], 'BCE': [], 'Loss': [],
                           'MSE': [], 'KL_output': [], 'tMSE': [],
                           'wMSE': []}
        self.test_loss = {'KL_latent': [], 'BCE': [], 'Loss': [],
                          'MSE': [], 'KL_output': [], 'tMSE': [],
                          'wMSE': []}
        self.num_steps = 0
        self.print_every = print_every
        #self.bce_loss = nn.BCELoss(reduction='mean')
        #self.mse_loss = nn.MSELoss(reduction='mean')
        #self.mae_loss = nn.L1Loss(reduction='mean')
        #self.kld_loss = nn.KLDivLoss(reduction='mean')
        #self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.beta = beta
        self.wb = wandb


    def _beta_scheduler(self, epoch, beta0=0., step=50, gamma=0.1):
        """Scheduler for beta value, the sheduler is a step function that
        increases beta value after "step" number of epochs by a factor "gamma"

        Parameters
        ----------
        epoch : int
            epoch value
        beta0 : float
            starting beta value
        step  : int
            epoch step for update
        gamma : float
            linear factor of step scheduler
            
        Returns
        -------
        beta
            beta value
        """
        
        if self.beta == 'step':
            return beta0 + gamma * (epoch // step)
        else:
            return float(self.beta)


    def _loss(self, x, xhat, mu, logvar, train=True, ep=0):
        """Evaluates loss function and add reports to the logger.
        Loss function is weighted MSe + KL divergeance. Also BCE 
        is calculate for comparison.

        Parameters
        ----------
        x      : tensor
            tensor of real values
        xhat   : tensor
            tensor of predicted values
        mu     : tensor
            tensor of mean values
        logvar : tensor
            tensor of log vairance values
        train  : bool
            wheather is training step or not
        ep     : int
            epoch value of training loop
            
        Returns
        -------
        loss
            loss value
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
        bce = F.binary_cross_entropy(xhat[:,:,0], x[:,:,0], reduction='mean')
        mse = F.mse_loss(xhat[:,:,0], x[:,:,0], reduction='mean')
        ## weighted mse
        #mse = torch.sum((xhat[:,:,0] - x[:,:,0]) ** 2 / x[:,:,-1]**2) / \
        #       (x.shape[0] * x.shape[1])
        
        kld_l = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0] / 1e4
        #kld_o = -1. * F.kl_div(xhat[:,:,-1], x[:,:,-1], reduction='mean')
        ## use MSE converge faster and better
        loss = mse + self._beta_scheduler(ep) * kld_l #+ 1 * kld_o
        

        if train:
            self.train_loss['BCE'].append(bce.item())# / (x.shape[0] * x.shape[1]))
            self.train_loss['MSE'].append(mse.item())# / (x.shape[0] * x.shape[1]))
            self.train_loss['KL_latent'].append(kld_l.item())# / (x.shape[0] * x.shape[1]))
            #self.train_loss['KL_output'].append(kld_o.item())
            #self.train_loss['wMSE'].append(wmse.item())
            self.train_loss['Loss'].append(loss.item())# / (x.shape[0] * x.shape[1]))
        else:
            self.test_loss['BCE'].append(bce.item())# / (x.shape[0] * x.shape[1]))
            self.test_loss['MSE'].append(mse.item())# / (x.shape[0] * x.shape[1]))
            self.test_loss['KL_latent'].append(kld_l.item())# / (x.shape[0] * x.shape[1]))
            #self.test_loss['KL_output'].append(kld_o.item())
            #self.test_loss['wMSE'].append(wmse.item())
            self.test_loss['Loss'].append(loss.item())# / (x.shape[0] * x.shape[1]))

        return loss


    ## function that does the in-epoch training
    def _train_epoch(self, data_loader, epoch):
        """Training loop for a given epoch. Triningo goes over 
        batches, light curves and latent space plots are logged to 
        W&B logger

        Parameters
        ----------
        data_loader : pytorch object
            data loader object with training items
        epoch       : int
            epoch number
            
        Returns
        -------

        """
        self.model.train()
        ## iterate over len(data)/batch_size
        mu_ep, labels = [], []
        xhat_plot, x_plot, l_plot = [], [], []
        for i, (data, label, onehot, pp) in enumerate(data_loader):

            self.num_steps += 1
            data = data.to(self.device)
            onehot = onehot.to(self.device)
            pp = pp.to(self.device)
            self.opt.zero_grad()

            if self.cond_l and not self.cond_p:
                xhat, mu, logvar, z = self.model(data, label=onehot)
            elif self.cond_p and not self.cond_l:
                xhat, mu, logvar, z = self.model(data, phy=pp)
            elif self.cond_l and self.cond_p:
                xhat, mu, logvar, z = self.model(data, label=onehot, phy=pp)
            else:
                xhat, mu, logvar, z = self.model(data)

            loss = self._loss(data[:,:,1:], xhat, 
                              mu, logvar, train=True, ep=epoch)
            loss.backward()
            self.opt.step()

            self._report_train(i)
            
            # concat time to recon
            xhat = torch.cat([data[:,:,0].unsqueeze(-1), xhat], dim=-1)
            # save arrays for plots
            mu_ep.append(mu.data.cpu().numpy())
            labels.extend(label)
            if epoch % 2== 0 and i == len(data_loader) - 2:
                xhat_plot = xhat.data.cpu().numpy()
                x_plot = data.data.cpu().numpy()
                l_plot = label

        mu_ep = np.concatenate(mu_ep)
        labels = np.array(labels)

        if epoch % 2 == 0:
            print(xhat_plot.shape)
            lc_wall, gif = plot_wall_time_series(xhat_plot,
                                                 cls=l_plot,
                                                 data_real=x_plot,
                                                 color='royalblue',
                                                 dim=(2, 4), figsize=(16, 4),
                                                 title='epoch %i' % epoch)
            self.wb.log({'Train_Recon_LCs':  self.wb.Image(lc_wall)},
                        step=self.num_steps)

        if epoch % 5 == 0:
            pp, image = plot_latent_space(mu_ep, y=labels)
            self.wb.log({'Latent_space_mu': self.wb.Image(pp)}, 
                        step=self.num_steps)


    def _test_epoch(self, test_loader, epoch):
        """Testing loop for a given epoch. Triningo goes over 
        batches, light curves and latent space plots are logged to 
        W&B logger

        Parameters
        ----------
        data_loader : pytorch object
            data loader object with training items
        epoch       : int
            epoch number
            
        Returns
        -------

        """
        self.model.eval()
        with torch.no_grad():
            xhat_plot, x_plot, l_plot = [], [], []
            for i, (data, label, onehot, pp) in enumerate(test_loader):
                data = data.to(self.device)
                onehot = onehot.to(self.device)
                pp = pp.to(self.device)

                if self.cond_l and not self.cond_p:
                    xhat, mu, logvar, z = self.model(data, label=onehot)
                elif self.cond_p and not self.cond_l:
                    xhat, mu, logvar, z = self.model(data, phy=pp)
                elif self.cond_l and self.cond_p:
                    xhat, mu, logvar, z = self.model(data, label=onehot, phy=pp)
                else:
                    xhat, mu, logvar, z = self.model(data)

                loss = self._loss(data[:,:,1:], xhat, 
                                  mu, logvar, train=False, ep=epoch)
                # concat time to recon
                xhat = torch.cat([data[:,:,0].unsqueeze(-1), xhat], dim=-1)
                # keep array for plots 
                if epoch % 2 == 0 and i == len(test_loader) - 2:
                    xhat_plot = xhat.data.cpu().numpy()
                    x_plot = data.data.cpu().numpy()
                    l_plot = label

        self._report_test(epoch)

        ## generate data with G for visualization and seve to tensorboard
        if epoch % 2 == 0:
            lc_wall, gif = plot_wall_time_series(xhat_plot,
                                                 cls=l_plot,
                                                 data_real=x_plot,
                                                 color='royalblue',
                                                 dim=(2, 4), figsize=(16, 4),
                                                 title='epoch %i' % epoch)
            self.wb.log({'Test_Recon_LCs':  self.wb.Image(lc_wall)},
                        step=self.num_steps)

        return loss



    def train(self, train_loader, test_loader, epochs, data_ex,
              machine='local', save=True, early_stop=False):
        """Full training loop over all epochs. Model is saved after 
        training is finished.

        Parameters
        ----------
        train_loader : pytorch object
            data loader object with training items
        test_loader : pytorch object
            data loader object with training items
        epoch       : int
            epoch number
        data_ex     : array
            array of example light curves
        machine     : str
            string with name of runing machine
        save        : bool
            wheather to save or not final model
        early_stop  : bool
            whaether to use early stop callback which stops training
            when validation loss converges
        Returns
        -------

        """

        ## hold samples, real and generated, for initial plotting
        if early_stop:
            early_stopping = EarlyStopping(patience=10, min_delta=.01,
                                           verbose=True)
        real_data_ex = data_ex[0]
        real_label_ex = data_ex[1]

        ## train for n number of epochs
        time_start = datetime.datetime.now()
        for epoch in range(1, epochs + 1):
            e_time = datetime.datetime.now()
            print('##'*20)
            print("\nEpoch {}".format(epoch))
            print("beta: %.2f" % self._beta_scheduler(epoch))

            # train and validate
            self._train_epoch(train_loader, epoch)
            val_loss = self._test_epoch(test_loader, epoch)

            # update learning rate according to cheduler
            if self.sch is not None:
                self.wb.log({'LR': self.opt.param_groups[0]['lr']},
                            step=self.num_steps)
                if 'ReduceLROnPlateau' == self.sch.__class__.__name__:
                    self.sch.step(val_loss)
                else:
                    self.sch.step(epoch)

            # report elapsed time per epoch and total run tume
            epoch_time = datetime.datetime.now() - e_time
            elap_time = datetime.datetime.now() - time_start
            print('Time per epoch: ', epoch_time.seconds, ' s')
            print('Elapsed time  : %.2f m' % (elap_time.seconds/60))
            print('##'*20)

            # early stopping
            if early_stop:
                early_stopping(val_loss.cpu())
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

        if save:
            torch.save(self.model.state_dict(), 
                       '%s/VAE_model_%s.pt' % 
                       (self.wb.run.dir, self.wb.run.name))


    def _report_train(self, i):
         """Report training metrics to logger and standard output.

        Parameters
        ----------
        i : int
            training step
            
        Returns
        -------

        """
        ## ------------------------ Reports ---------------------------- ##
        ## print scalars to std output and save scalars/hist to W&B
        if i % self.print_every == 0:
            print("Training iteration %i, global step %i" %
                  (i + 1, self.num_steps))
            print("BCE : %3.4f" % (self.train_loss['BCE'][-1]))
            print("MSE : %3.4f" % (self.train_loss['MSE'][-1]))
            print("KL_l: %3.4f" % (self.train_loss['KL_latent'][-1]))
            print("Loss: %3.4f" % (self.train_loss['Loss'][-1]))

            self.wb.log({
                'Train_BCE'      : self.train_loss['BCE'][-1],
                'Train_MSE'      : self.train_loss['MSE'][-1],
                'Train_KL_latent': self.train_loss['KL_latent'][-1],
                #'Train_KL_output': self.train_loss['KL_output'][-1],
                #'Train_tMSE'      : self.train_loss['tMSE'][-1],
                #'Train_wMSE'      : self.train_loss['wMSE'][-1],
                'Train_Loss'     : self.train_loss['Loss'][-1]},
                        step=self.num_steps)
            print("__"*20)

    def _report_test(self, ep):
        """Report testing metrics to logger and standard output.

        Parameters
        ----------
        i : int
            training step
            
        Returns
        -------

        """
        ## ------------------------ Reports ---------------------------- ##
        ## print scalars to std output and save scalars/hist to W&B
        print('*** TEST LOSS ***')
        print("Epoch %i, global step %i" % (ep, self.num_steps))
        print("BCE : %3.4f" % (self.test_loss['BCE'][-1]))
        print("MSE : %3.4f" % (self.test_loss['MSE'][-1]))
        print("KL_l: %3.4f" % (self.test_loss['KL_latent'][-1]))
        print("Loss: %3.4f" % (self.test_loss['Loss'][-1]))

        self.wb.log({
            'Test_BCE'      : self.test_loss['BCE'][-1],
            'Test_MSE'      : self.test_loss['MSE'][-1],
            'Test_KL_latent': self.test_loss['KL_latent'][-1],
            #'Test_KL_output': self.test_loss['KL_output'][-1],
            #'Test_tMSE'      : self.test_loss['tMSE'][-1],
            #'Test_wMSE'      : self.test_loss['wMSE'][-1],
            'Test_Loss'     : self.test_loss['Loss'][-1]},
                    step=self.num_steps)
        print("__"*20)
