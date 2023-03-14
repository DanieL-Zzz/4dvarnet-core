from hydra.core.config_store import ConfigStore
from itertools import product

cs = ConfigStore.instance()

experiment_prefix_name = 'MP'

BASE_DEFAULTS = [
    '/xp/baseline/train_strat/const_lr_ngrad_5_3cas',
    '/splits/dc_boost_swot@datamodule',
]

n_priors = 2


# 1. model
# --------

_models = {
	'4dvarnet_OI': {
        'node': {'model': '4dvarnet_OI'},
        'group': 'model',
        'package': 'params',
    },

    'multiprior': {
        'node': {
            'params': {'model': 'multiprior', 'nb_phi': n_priors},
            'datamodule':{
                'dataset_class': {
                    '_target_': 'dataloading.FourDVarNetDatasetLatLon',
                    '_partial_': True
                }
            }
        },
        'group': 'model',
        'package': '_global_',
    }
}

model = {key: f'/model/{key}' for key in _models}


# 2. aug train data
# -----------------

_aug_train_data = {
	'1': {
        'node': {'aug_train_data': True},
        'group': 'aug_data',
        'package': 'datamodule',
    },
}

aug = {f'aug{key}': f'/aug_data/{key}' for key in _aug_train_data}


# 3. resize_factor
# ----------------

_resize_factor = {
	'4': {
        'node': {'resize_factor': 4},
        'group': 'down_samp',
        'package': 'datamodule',
    }
}

resize = {f'ds{key}': f'/down_samp/{key}' for key in _resize_factor}


# 4. patch_weight
# ---------------

_patch_weight = {
	'29_8': {
        'node': {
            'patch_weight': {
                '_target_': 'lit_model_augstate.get_constant_crop',
                'patch_size': '${datamodule.slice_win}',
                'crop': {'time': 8, 'lat': '${div:20,${datamodule.resize_factor}}', 'lon': '${div:20,${datamodule.resize_factor}}'}
            },
	        'dT': 29,
        },
        'group': 'dT',
        'package': 'params',
    }
}

dT = {f'dT{key}': f'/dT/{key}' for key in _patch_weight}


# 5. store the configs
# --------------------

dicts = _models | _aug_train_data | _resize_factor | _patch_weight
for key in dicts:
	cs.store(
		name=key,
		node=dicts[key]['node'],
		group=dicts[key]['group'],
		package=dicts[key]['package'],
    )


# 6. combine
# ----------

combinations = product(
    [
        ('no_sst', '/xp/qfebvre/xp_oi_cnatl'),
    ],

    # training and test areas format trainArea_testArea
    [
        ('gf_10x20', '/xp/baseline/dl/d240_p240x5_s200x1_10x20'),
    ],

    [
        ('swot_4nad', '/xp/qfebvre/ds/swot_four_nadirs_dc.yaml'),
    ],
    model.items(),
    aug.items(),
    resize.items(),
    dT.items(),
)

for defaults in combinations:
    labels, defaults = zip(*defaults)
    defaults_xp_name = '_'.join(labels)

    xp_name = f'{experiment_prefix_name}_{defaults_xp_name}'
    cfg = {
        'xp_name': xp_name,
        'defaults': BASE_DEFAULTS + list(defaults) + ['_self_'],
    }

    cs.store(name=xp_name, node=cfg, group='xp', package='_global_')


if __name__== '__main__':
	for xp  in cs.list('xp'):
		print(xp)

