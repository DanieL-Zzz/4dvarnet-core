import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

import hydra
import pandas as pd
from datetime import datetime, timedelta
from hydra.utils import get_class, instantiate, call
from omegaconf import OmegaConf
import hydra_config
import numpy as np
import hashlib
import json
from dashtable import data2rst


def _hash(_model):
    d = _model.state_dict()
    for key, value in d.items():
        if isinstance(value, torch.Tensor):
            d[key] = value.cpu().numpy().tolist()
    return hashlib.sha256(
        json.dumps(d, sort_keys=True).encode('utf8')
    ).hexdigest()[:10]


def _hit(_model):  # hash, id, type
    return _hash(_model), str(id(_model))[-10:], type(_model)


def get_profiler():
    from pytorch_lightning.profiler import PyTorchProfiler
    print( torch.profiler.ProfilerActivity.CPU,)
    return PyTorchProfiler(
            "results/profile_report",
            schedule=torch.profiler.schedule(
                skip_first=2,
                wait=2,
                warmup=2,
                active=2),
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            # with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./tb_profile'),
            record_shapes=True,
            profile_memory=True,
    )


class FourDVarNetHydraRunner:
    def __init__(self, params, dm, lit_mod_cls, callbacks=None, logger=None):
        self.cfg = params
        self.filename_chkpt = self.cfg.ckpt_name
        self.callbacks = callbacks
        self.logger = logger
        self.dm = dm
        self.lit_cls = lit_mod_cls
        dm.setup()
        self.dataloaders = {
            'train': dm.train_dataloader(),
            'val': dm.val_dataloader(),
            'test': dm.test_dataloader(),
        }

        test_dates = np.concatenate([ \
                       [str(dt.date()) for dt in \
                       pd.date_range(dm.test_slices[i].start,dm.test_slices[i].stop)[(self.cfg.dT//2):-(self.cfg.dT//2)]] \
                      for i in range(len(dm.test_slices))])
        #print(test_dates)
        self.time = {'time_test' : test_dates}

        self.setup(dm)

    def setup(self, datamodule):
        self.mean_Tr = datamodule.norm_stats[0]
        self.mean_Tt = datamodule.norm_stats[0]
        self.mean_Val = datamodule.norm_stats[0]
        self.var_Tr = datamodule.norm_stats[1] ** 2
        self.var_Tt = datamodule.norm_stats[1] ** 2
        self.var_Val = datamodule.norm_stats[1] ** 2
        self.min_lon = datamodule.dim_range['lon'].start
        self.max_lon = datamodule.dim_range['lon'].stop
        self.min_lat = datamodule.dim_range['lat'].start
        self.max_lat = datamodule.dim_range['lat'].stop
        self.ds_size_time = datamodule.ds_size['time']
        self.ds_size_lon = datamodule.ds_size['lon']
        self.ds_size_lat = datamodule.ds_size['lat']
        self.dX = int((datamodule.slice_win['lon']-datamodule.strides['lon'])/2)
        self.dY = int((datamodule.slice_win['lat']-datamodule.strides['lat'])/2)
        self.swX = datamodule.slice_win['lon']
        self.swY = datamodule.slice_win['lat']
        self.lon, self.lat = datamodule.coordXY()
        w_ = np.zeros(self.cfg.dT)
        w_[int(self.cfg.dT / 2)] = 1.
        self.wLoss = torch.Tensor(w_)
        self.resolution = datamodule.resolution
        self.original_coords = datamodule.get_original_coords()
        self.padded_coords = datamodule.get_padded_coords()

    def run(self, ckpt_path=None, dataloader="test", **trainer_kwargs):
        """
        Train and test model and run the test suite
        :param ckpt_path: (Optional) Checkpoint from which to resume
        :param dataloader: Dataloader on which to run the test Checkpoint from which to resume
        :param trainer_kwargs: (Optional)
        """
        mod, trainer = self.train(ckpt_path, **trainer_kwargs)
        self.test(dataloader=dataloader, _mod=mod, _trainer=trainer)

    def _get_model(self, ckpt_path=None):
        """
        Load model from ckpt_path or instantiate new model
        :param ckpt_path: (Optional) Checkpoint path to load
        :return: lightning module
        """
        print('get_model: ', ckpt_path)
        if ckpt_path:
            mod = self.lit_cls.load_from_checkpoint(ckpt_path,
                                                    hparam=self.cfg,
                                                    w_loss=self.wLoss,
                                                    strict=False,
                                                    mean_Tr=self.mean_Tr,
                                                    mean_Tt=self.mean_Tt,
                                                    mean_Val=self.mean_Val,
                                                    var_Tr=self.var_Tr,
                                                    var_Tt=self.var_Tt,
                                                    var_Val=self.var_Val,
                                                    min_lon=self.min_lon, max_lon=self.max_lon,
                                                    min_lat=self.min_lat, max_lat=self.max_lat,
                                                    ds_size_time=self.ds_size_time,
                                                    ds_size_lon=self.ds_size_lon,
                                                    ds_size_lat=self.ds_size_lat,
                                                    time=self.time,
                                                    dX=self.dX, dY = self.dY,
                                                    swX=self.swX, swY=self.swY,
                                                    coord_ext={'lon_ext': self.lon,
                                                               'lat_ext': self.lat},
                                                    test_domain=self.cfg.test_domain,
                                                    resolution=self.resolution,
                                                    original_coords=self.original_coords,
                                                    padded_coords=self.padded_coords
                                                    )

        else:
            mod = self.lit_cls(hparam=self.cfg,
                               w_loss=self.wLoss,
                               mean_Tr=self.mean_Tr,
                               mean_Tt=self.mean_Tt,
                               mean_Val=self.mean_Val,
                               var_Tr=self.var_Tr,
                               var_Tt=self.var_Tt,
                               var_Val=self.var_Val,
                               min_lon=self.min_lon, max_lon=self.max_lon,
                               min_lat=self.min_lat, max_lat=self.max_lat,
                               ds_size_time=self.ds_size_time,
                               ds_size_lon=self.ds_size_lon,
                               ds_size_lat=self.ds_size_lat,
                               time=self.time,
                               dX=self.dX, dY = self.dY,
                               swX=self.swX, swY=self.swY,
                               coord_ext = {'lon_ext': self.lon,
                                            'lat_ext': self.lat},
                               test_domain=self.cfg.test_domain,
                               resolution=self.resolution,
                               original_coords=self.original_coords,
                               padded_coords=self.padded_coords
                               )
        return mod

    def _inject_OI_to_MP(self, mod, ckpt_path):
        _mod = self.lit_cls.load_from_checkpoint(
            ckpt_path,
            hparam=self.cfg,
            strict=False,
            test_domain=self.cfg.test_domain,
            mean_Tr=self.mean_Tr,
            mean_Tt=self.mean_Tt,
            mean_Val=self.mean_Val,
            var_Tr=self.var_Tr,
            var_Tt=self.var_Tt,
            var_Val=self.var_Val,
        )

        print('\n---- IN _inject_OI_to_MP ----\n')

        print('>>> ckpt_path:', ckpt_path, '\n')

        _headers = [
            '',
            'mod (hash)', 'mod (id)', 'mod (type)',
            '_mod (hash)', '_mod (id)', '_mod (type)'
        ]

        print('>>> 0. General')
        print(data2rst(
            [
                _headers,
                ['.', *_hit(mod), *_hit(_mod)],
                [
                    '.model',
                    *_hit(mod.model),
                    *_hit(_mod.model)
                ],
                [
                    '.model.phi_r',
                    *_hit(mod.model.phi_r),
                    *_hit(_mod.model.phi_r)
                ],
            ],
            use_headers=True,
        ))

        print('>>> 1. Injecting _mod into mod')
        print(data2rst(
            [
                _headers,
                [
                    '.model.model_H',
                    *_hit(mod.model.model_H),
                    *_hit(_mod.model.model_H)
                ],
                [
                    '.model.model_Grad',
                    *_hit(mod.model.model_Grad),
                    *_hit(_mod.model.model_Grad)
                ],
                [
                    '.model.model_VarCost',
                    *_hit(mod.model.model_VarCost),
                    *_hit(_mod.model.model_VarCost)
                ],
            ],
            use_headers=True,
        ))
        mod.model.model_H.load_state_dict(_mod.model.model_H.state_dict())
        mod.model.model_Grad.load_state_dict(_mod.model.model_Grad.state_dict())
        mod.model.model_VarCost.load_state_dict(_mod.model.model_VarCost.state_dict())
        print('>>> after transfer:')
        print(data2rst(
            [
                _headers,
                [
                    '.model.model_H',
                    *_hit(mod.model.model_H),
                    *_hit(_mod.model.model_H)
                ],
                [
                    '.model.model_Grad',
                    *_hit(mod.model.model_Grad),
                    *_hit(_mod.model.model_Grad)
                ],
                [
                    '.model.model_VarCost',
                    *_hit(mod.model.model_VarCost),
                    *_hit(_mod.model.model_VarCost)
                ],
            ],
            use_headers=True,
        ))

        print(">>> 2. Injecting _mod's phi into mod's priors[0]")
        _phis = []
        for i in range(len(mod.model.phi_r.priors)):
            cphi = mod.model.phi_r.priors[i]
            rowlegend = f'.model.phi_r.priors[{i}]'

            _phis.append([
                rowlegend,
                *_hit(cphi),
                *_hit(_mod.model.phi_r)
            ])

            # transfer
            mod.model.phi_r.priors[i].load_state_dict(_mod.model.phi_r.state_dict())

        # Noise last prior's last layer (N(0, 10⁻²))
        # mod.model.phi_r.priors[-1].noise_last_layer(0., .01)

        print(data2rst(
            [_headers, *_phis],
            use_headers=True,
        ))

        print('>>> after transfer:')
        _phis = []
        for i in range(len(mod.model.phi_r.priors)):
            cphi = mod.model.phi_r.priors[i]
            rowlegend = f'.model.phi_r.priors[{i}]'

            _phis.append([
                rowlegend,
                *_hit(cphi),
                *_hit(_mod.model.phi_r)
            ])

        print(data2rst(
            [_headers, *_phis],
            use_headers=True,
        ))

        print(mod)

        print('>>> LEAVING _turn_OI_to_MP')

    def train(self, ckpt_path=None, **trainer_kwargs):
        """
        Train a model
        :param ckpt_path: (Optional) Checkpoint from which to resume
        :param trainer_kwargs: (Optional) Trainer arguments
        :return:
        """

        mod = self._get_model(ckpt_path=ckpt_path)

        qmodel_path = trainer_kwargs.pop('qmodel_path', None)
        if qmodel_path:
            self._inject_OI_to_MP(mod, qmodel_path)

        checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                              filename=self.filename_chkpt,
                                              save_top_k=3,
                                              mode='min')
        from pytorch_lightning.callbacks import LearningRateMonitor
        lr_monitor = LearningRateMonitor(logging_interval='step')
        num_nodes = int(os.environ.get('SLURM_JOB_NUM_NODES', 1))
        gpus = trainer_kwargs.get('gpus', torch.cuda.device_count())

        num_gpus = gpus if isinstance(gpus, (int, float)) else  len(gpus) if hasattr(gpus, '__len__') else 0
        accelerator = "ddp" if (num_gpus * num_nodes) > 1 else None
        trainer_kwargs_final = {**dict(num_nodes=num_nodes, gpus=gpus, logger=self.logger, strategy=accelerator, auto_select_gpus=(num_gpus * num_nodes) > 0,
                             callbacks=[checkpoint_callback, lr_monitor]),  **trainer_kwargs}
        print(trainer_kwargs)
        print(trainer_kwargs_final)
        trainer = pl.Trainer(**trainer_kwargs_final)
        trainer.fit(mod, self.dataloaders['train'], self.dataloaders['val'])
        return mod, trainer

    def test(self, ckpt_path=None, dataloader="test", _mod=None, _trainer=None, **trainer_kwargs):
        """
        Test a model
        :param ckpt_path: (Optional) Checkpoint from which to resume
        :param dataloader: Dataloader on which to run the test Checkpoint from which to resume
        :param trainer_kwargs: (Optional)
        """
        trainer_kwargs.pop('qmodel_path', None)
        trainer = _trainer or pl.Trainer(
            num_nodes=1, gpus=1, accelerator=None, **trainer_kwargs,
        )
        mod = _mod or self._get_model(ckpt_path=ckpt_path)

        trainer.test(mod, dataloaders=self.dataloaders[dataloader])
        return mod

    def profile(self):
        """
        Run the profiling
        :return:
        """
        from pytorch_lightning.profiler import PyTorchProfiler

        profiler = PyTorchProfiler(
            "results/profile_report",
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=1),
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./tb_profile'),
            record_shapes=True,
            profile_memory=True,
        )
        self.train(
            **{
                'profiler': profiler,
                'max_epochs': 1,
            }
        )


def _main(cfg):
    print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(seed=cfg.get('seed', None))
    dm = instantiate(cfg.datamodule)
    if cfg.get('callbacks') is not None:
        callbacks = [instantiate(cb_cfg) for cb_cfg in cfg.callbacks]
    else:
        callbacks=[]

    if cfg.get('logger') is not None:
        print('instantiating logger')
        print(OmegaConf.to_yaml(cfg.logger))
        logger = instantiate(cfg.logger)
    else:
        logger=True
    lit_mod_cls = get_class(cfg.lit_mod_cls)
    runner = FourDVarNetHydraRunner(cfg.params, dm, lit_mod_cls, callbacks=callbacks, logger=logger)
    call(cfg.entrypoint, self=runner)


main = hydra.main(config_path='hydra_config', config_name='main')(_main)

if __name__ == '__main__':
    main()
