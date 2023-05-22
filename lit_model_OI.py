import hydra
import pandas as pd
from torch import nn
import xarray as xr
from pathlib import Path
from hydra.utils import call
import numpy as np
import torch
import solver as NN_4DVar
import metrics
from metrics import plot_maps_oi
from models import Model_H, Phi_r_OI, Multi_Prior, Lat_Lon_Multi_Prior
from lit_model_augstate import LitModelAugstate


def get_4dvarnet_OI(hparams):
    return NN_4DVar.Solver_Grad_4DVarNN(
                Phi_r_OI(hparams.shape_state[0], hparams.DimAE, hparams.dW, hparams.dW2, hparams.sS,
                    hparams.nbBlocks, hparams.dropout_phi_r, hparams.stochastic),
                Model_H(hparams.shape_state[0]),
                NN_4DVar.model_GradUpdateLSTM(hparams.shape_state, hparams.UsePriofepodicBoundary,
                    hparams.dim_grad_solver, hparams.dropout),
                hparams.norm_obs, hparams.norm_prior, hparams.shape_state, hparams.n_grad * hparams.n_fourdvar_iter)


def get_multi_prior(hparams):
    return NN_4DVar.Solver_Grad_4DVarNN(
                Multi_Prior(hparams.shape_state, hparams.DimAE, hparams.dW, hparams.dW2, hparams.sS,
                    hparams.nbBlocks, hparams.dropout_phi_r, hparams.nb_phi, hparams.stochastic),
                Model_H(hparams.shape_state[0]),
                NN_4DVar.model_GradUpdateLSTM(hparams.shape_state, hparams.UsePriodicBoundary,
                    hparams.dim_grad_solver, hparams.dropout),
                hparams.norm_obs, hparams.norm_prior, hparams.shape_state, hparams.n_grad * hparams.n_fourdvar_iter)


def get_lat_lon_multi_prior(hparams):
    return NN_4DVar.Solver_Grad_4DVarNN_Lat_Lon(
                Lat_Lon_Multi_Prior(hparams.shape_state, hparams.DimAE, hparams.dW, hparams.dW2, hparams.sS,
                    hparams.nbBlocks, hparams.dropout_phi_r, hparams.nb_phi, hparams.stochastic),
                Model_H(hparams.shape_state[0]),
                NN_4DVar.model_GradUpdateLSTM(hparams.shape_state, hparams.UsePriodicBoundary,
                    hparams.dim_grad_solver, hparams.dropout),
                hparams.norm_obs, hparams.norm_prior, hparams.shape_state, hparams.n_grad * hparams.n_fourdvar_iter)


class LitModelOI(LitModelAugstate):
    MODELS = {
        '4dvarnet_OI': get_4dvarnet_OI,
        'multi_prior': get_multi_prior,
        'lat_lon_multi_prior': get_lat_lon_multi_prior
     }

    def configure_optimizers(self):
        opt = torch.optim.Adam

        if hasattr(self.hparams, 'opt'):
            opt = lambda p: hydra.utils.call(self.hparams.opt, p)

        _lr = self.hparams.lr_update[0]
        optimizer = opt([
            {'params': self.model.model_Grad.parameters(), 'lr': _lr},
            {'params': self.model.model_VarCost.parameters(), 'lr': _lr},
            {'params': self.model.model_H.parameters(), 'lr': _lr},
            {'params': self.model.phi_r.parameters(), 'lr': 0.5*_lr},
        ])

        _lr_scheduler = self.hparams.get('lr_scheduler', None)
        if _lr_scheduler:
            print('>>> using a learning scheduler:', _lr_scheduler)
            _partial = hydra.utils.instantiate(_lr_scheduler)
            lr_scheduler = _partial(optimizer=optimizer)

            return {
                'optimizer': optimizer,
                'lr_scheduler': lr_scheduler,
            }
        return optimizer

    def diag_step(self, batch, batch_idx, log_pref='test'):
        oi, inputs_Mask, inputs_obs, targets_GT, *_= batch
        losses, out, metrics = self(batch, phase='test')
        loss = losses[-1]

        if torch.isnan(loss):
            raise Exception(f'Loss is nan')

        if loss is not None and log_pref is not None:
            self.log(f'{log_pref}_loss', loss)
            self.log(
                f'{log_pref}_mse',
                metrics[-1]["mse"] / self.var_Tt,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                f'{log_pref}_mseG',
                metrics[-1]['mseGrad'] / metrics[-1]['meanGrad'],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        # Destandardise function
        _d = lambda X: X.detach().cpu() * np.sqrt(self.var_Tr) + self.mean_Tr

        if self.model_name in ['multi_prior', 'lat_lon_multi_prior']:
            if self.model_name == 'lat_lon_multi_prior':
                oi, inputs_Mask, inputs_obs, targets_GT, latitude, longitude, *_ = batch
                results, weights = self.model.phi_r.get_intermediate_results(
                    out.detach(), latitude, longitude,
                )
            else:
                results, weights = self.model.phi_r.get_intermediate_results(out.detach())

            return {
                'gt': _d(targets_GT),
                'obs_inp': _d(
                    inputs_obs
                    .where(inputs_Mask, torch.full_like(inputs_obs, np.nan))
                ),
                'oi': _d(oi),
                'pred': _d(out),
                **results,
                **weights,
            }

        return {
            'gt': _d(targets_GT),
            'obs_inp': _d(
                inputs_obs
                .where(inputs_Mask, torch.full_like(inputs_obs, np.nan))
            ),
            'oi': _d(oi),
            'pred': _d(out),
        }


    def sla_diag(self, t_idx=3, log_pref='test'):
        path_save0 = self.logger.log_dir + f'/{log_pref}_maps.png'
        fig_maps = plot_maps_oi(
                  self.x_gt[t_idx],
                self.obs_inp[t_idx],
                  self.x_rec[t_idx],
                  self.test_lon, self.test_lat, path_save0)
        path_save01 = self.logger.log_dir + f'/{log_pref}_maps_Grad.png'
        fig_maps_grad = plot_maps_oi(
                  self.x_gt[t_idx],
                self.obs_inp[t_idx],
                  self.x_rec[t_idx],
                  self.test_lon, self.test_lat, path_save01, grad=True)
        self.test_figs['maps'] = fig_maps
        self.test_figs['maps_grad'] = fig_maps_grad
        self.logger.experiment.add_figure(f'{log_pref} Maps', fig_maps, global_step=self.current_epoch)
        self.logger.experiment.add_figure(f'{log_pref} Maps Grad', fig_maps_grad, global_step=self.current_epoch)

        lamb_x, lamb_t, mu, sig = np.nan, np.nan, np.nan, np.nan
        try:
            psd_ds, lamb_x, lamb_t = metrics.psd_based_scores(self.test_xr_ds.pred, self.test_xr_ds.gt)
            psd_fig = metrics.plot_psd_score(psd_ds)
            self.test_figs['psd'] = psd_fig
            self.logger.experiment.add_figure(f'{log_pref} PSD', psd_fig, global_step=self.current_epoch)
            _, _, mu, sig = metrics.rmse_based_scores(self.test_xr_ds.pred, self.test_xr_ds.gt)
        except:
            print('fail to compute psd scores')
        mse_metrics_pred = metrics.compute_metrics(self.test_xr_ds.gt, self.test_xr_ds.pred)
        mse_metrics_oi = metrics.compute_metrics(self.test_xr_ds.gt, self.test_xr_ds.oi)
        var_mse_pred_vs_oi = 100. * ( 1. - mse_metrics_pred['mse'] / mse_metrics_oi['mse'] )
        var_mse_grad_pred_vs_oi = 100. * ( 1. - mse_metrics_pred['mseGrad'] / mse_metrics_oi['mseGrad'] )
        md = {
            f'{log_pref}_lambda_x': lamb_x,
            f'{log_pref}_lambda_t': lamb_t,
            f'{log_pref}_mu': mu,
            f'{log_pref}_sigma': sig,
            f'{log_pref}_var_mse_vs_oi': float(var_mse_pred_vs_oi),
            f'{log_pref}_var_mse_grad_vs_oi': float(var_mse_grad_pred_vs_oi),
        }
        print(pd.DataFrame([md]).T.to_markdown())
        return md

    def diag_epoch_end(self, outputs, log_pref='test'):
        full_outputs = self.gather_outputs(outputs, log_pref=log_pref)
        if full_outputs is None:
            print("full_outputs is None on ", self.global_rank)
            return
        if log_pref == 'test':
            diag_ds = self.trainer.test_dataloaders[0].dataset.datasets[0]
        elif log_pref == 'val':
            diag_ds = self.trainer.val_dataloaders[0].dataset.datasets[0]
        else:
            raise Exception('unknown phase')
        self.test_xr_ds = self.build_test_xr_ds(full_outputs, diag_ds=diag_ds)
        log_path = Path(self.logger.log_dir).mkdir(exist_ok=True)
        print('########', f'{log_path=}')
        path_save1 = self.logger.log_dir + f'/{log_pref}.nc'
        self.test_xr_ds.to_netcdf(path_save1)

        self.x_gt = self.test_xr_ds.gt.data
        self.oi = self.test_xr_ds.oi.data
        self.obs_inp = self.test_xr_ds.obs_inp.data

        self.x_rec = self.test_xr_ds.pred.data
        self.x_rec_ssh = self.x_rec

        self.test_coords = self.test_xr_ds.coords
        self.test_lat = self.test_coords['lat'].data
        self.test_lon = self.test_coords['lon'].data
        self.test_dates = self.test_coords['time'].data

        # display map
        md = self.sla_diag(t_idx=2, log_pref=log_pref)

        self.latest_metrics.update(md)
        self.logger.log_metrics(md, step=self.current_epoch)

    def get_init_state(self, batch, state=(None,)):
        if state[0] is not None:
            return state[0]

        _, inputs_Mask, inputs_obs, *_ = batch

        init_state = inputs_Mask * inputs_obs
        return init_state

    def loss_ae(self, state_out, latitude=None, longitude=None):
        #Ignore autoencoder loss for fixed point solver
        if self.model_name in ['UNet_FP']:
            return 0.
        elif self.model_name in ['lat_lon_multi_prior']:
            return torch.mean((self.model.phi_r(state_out, latitude, longitude) - state_out) ** 2)
        else:
            #same as in lit_model_augstate
            return torch.mean((self.model.phi_r(state_out) - state_out) ** 2)

    def compute_loss(self, batch, phase, state_init=(None,)):
        if self.model_name in ['lat_lon_multi_prior']:
             #Latitude and longitude are used for the weight matrix for the multi-prior
            if self.use_sst:
                _, inputs_Mask, inputs_obs, targets_GT, latitude, longitude, sst_gt = batch

            else:
                _, inputs_Mask, inputs_obs, targets_GT, latitude, longitude = batch

        else:
            latitude = None
            longitude = None #Passed to loss AE
            if self.use_sst:
                _, inputs_Mask, inputs_obs, targets_GT, sst_gt = batch

            else:
                _, inputs_Mask, inputs_obs, targets_GT = batch



        # handle patch with no observation
        if inputs_Mask.sum().item() == 0:
            return (
                    None,
                    torch.zeros_like(targets_GT),
                    torch.cat((torch.zeros_like(targets_GT),
                              torch.zeros_like(targets_GT),
                              torch.zeros_like(targets_GT)), dim=1),
                    dict([('mse', 0.),
                        ('mseGrad', 0.),
                        ('meanGrad', 1.),
                        ])
                    )
        targets_GT_wo_nan = targets_GT.where(~targets_GT.isnan(), torch.zeros_like(targets_GT))

        state = self.get_init_state(batch, state_init)

        obs = inputs_Mask * inputs_obs
        new_masks =  inputs_Mask

        if self.use_sst:
            new_masks = [ new_masks, torch.ones_like(sst_gt) ]
            obs = [ obs, sst_gt ]

        # gradient norm field
        g_targets_GT_x, g_targets_GT_y = self.gradient_img(targets_GT)

        # need to evaluate grad/backward during the evaluation and training phase for phi_r
        with torch.set_grad_enabled(True):
            state = torch.autograd.Variable(state, requires_grad=True)
            if self.model_name in ['lat_lon_multi_prior']:
                outputs, hidden_new, cell_new, normgrad = self.model(state, obs, new_masks, latitude, longitude, *state_init[1:])
            else:
                outputs, hidden_new, cell_new, normgrad = self.model(state, obs, new_masks, *state_init[1:])

            if (phase == 'val') or (phase == 'test'):
                outputs = outputs.detach()

            loss_All, loss_GAll = self.sla_loss(outputs, targets_GT_wo_nan)
            loss_AE = self.loss_ae(outputs, latitude, longitude)

            # total loss
            loss = self.hparams.alpha_mse_ssh * loss_All + self.hparams.alpha_mse_gssh * loss_GAll
            loss += 0.5 * self.hparams.alpha_proj * loss_AE

            # metrics
            # mean_GAll = NN_4DVar.compute_spatio_temp_weighted_loss(g_targets_GT, self.w_loss)
            mean_GAll = NN_4DVar.compute_spatio_temp_weighted_loss(
                    torch.hypot(g_targets_GT_x, g_targets_GT_y) , self.grad_crop(self.patch_weight))
            mse = loss_All.detach()
            mseGrad = loss_GAll.detach()
            metrics = dict([
                ('mse', mse),
                ('mseGrad', mseGrad),
                ('meanGrad', mean_GAll),
                ])

        return loss, outputs, [outputs, hidden_new, cell_new, normgrad], metrics
