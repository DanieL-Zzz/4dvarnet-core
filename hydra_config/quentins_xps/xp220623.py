from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pathlib import Path
from itertools import product

cs = ConfigStore.instance()

BASE_DEFAULTS = [
  '/splits/dc_boost_swot@datamodule',
]
XP=23

cs.store(name='0', node={'aug_train_data': False}, group='aug_data', package='datamodule')
cs.store(name='1', node={'aug_train_data': True}, group='aug_data', package='datamodule')
cs.store(name='2', node={'aug_train_data': 2}, group='aug_data', package='datamodule')
cs.store(name='3', node={'aug_train_data': 3}, group='aug_data', package='datamodule')
cs.store(name='8', node={'aug_train_data': 8}, group='aug_data', package='datamodule')
aug = {
    # 'aug0': '/aug_data/0',
    # 'aug1': '/aug_data/1',
    # 'aug2': '/aug_data/2',
    'aug3': '/aug_data/3',
    'aug8': '/aug_data/8',
}

cs.store(name='2', node={'resize_factor': 2}, group='down_samp', package='datamodule')
cs.store(name='1', node={'resize_factor': 1}, group='down_samp', package='datamodule')
resize = {
    'ds2': '/down_samp/2',
    'ds1': '/down_samp/1',
}

cs.store(name='29_8', node={
    'patch_weight': {
        '_target_': 'lit_model_augstate.get_constant_crop',
        'patch_size': '${datamodule.slice_win}',
        'crop': { 'time': 8, 'lat': '${div:20,${datamodule.resize_factor}}', 'lon':  '${div:20,${datamodule.resize_factor}}'}
    }, 'dT': 29, }, group='dT', package='params')
cs.store(name='29_13', node={
    'patch_weight': {
        '_target_': 'lit_model_augstate.get_constant_crop',
        'patch_size': '${datamodule.slice_win}',
        'crop': { 'time': 13, 'lat': '${div:20,${datamodule.resize_factor}}', 'lon':  '${div:20,${datamodule.resize_factor}}'}
    }, 'dT': 29, }, group='dT', package='params')
dT = {
    'dT29_8': '/dT/29_8',
    'dT29_13': '/dT/29_13',
}

for  defaults in product(
        [
            ('sst', '/xp/qfebvre/xp_oi_sst'),
            ('no_sst', '/xp/qfebvre/xp_oi'),
        ],
        [
            # ('swot', '/xp/qfebvre/ds/clean_swot_oi_no_swot.yaml'),
            ('5nad', '/xp/qfebvre/ds/five_nadirs.yaml'),
            ('swot_w_oi', '/xp/qfebvre/ds/clean_swot_oi_w_swot.yaml'),
            # ('roll', '/xp/qfebvre/ds/new_noisy_swot_roll.yaml'),
            # ('syst', '/xp/qfebvre/ds/new_noisy_swot_syst_errs.yaml'),
            # ('noisy', '/xp/qfebvre/ds/new_noisy_swot.yaml'),
        ],
        aug.items(),
        resize.items(),
        dT.items(),
    ):
    labels, defaults = zip(*defaults)
    defaults_xp_name = '_'.join(labels)

    xp_name = f'qxp{XP}_' + defaults_xp_name 
    cfg = {
            'xp_name': xp_name,
            'defaults': BASE_DEFAULTS + list(defaults) + ['_self_'],
        }
    cs.store(name=xp_name, node=cfg, group='xp', package='_global_')
    # print(cs.list('xp'))


if __name__== '__main__':
	for xp  in cs.list('xp'):
		print(xp)

