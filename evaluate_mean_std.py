from argparse import ArgumentParser
from functools import partial
import json

from tqdm import trange
import numpy as np

from src.trainer import GalaxyZooInfoSCC_Trainer
from src.utils import get_config


def main(args):
    n = args.n  # number of runs to average over

    config = get_config(args.config)
    trainer = GalaxyZooInfoSCC_Trainer(args.config, config)

    eval_methods = {
        'FID IV3': trainer._compute_fid_score,
        'FID SSL': partial(trainer._compute_fid, encoder_type='simclr'),
        'FID AE': partial(trainer._compute_fid, encoder_type='ae'),

        'IS': trainer._compute_inception_score,

        'Chamfer_SSL': partial(trainer._compute_chamfer_distance, encoder_type='simclr'),
        'Chamfer_AE': partial(trainer._compute_chamfer_distance, encoder_type='ae'),

        'PPL SSL': partial(trainer._compute_ppl, encoder_type='simclr'),
        'PPL VGG': partial(trainer._compute_ppl, encoder_type='vgg'),
        'PPL AE': partial(trainer._compute_ppl, encoder_type='ae'),

        'KID IV3': partial(trainer._compute_kid, encoder_type='inception'),
        'KID SSL': partial(trainer._compute_kid, encoder_type='simclr'),
        'KID AE': partial(trainer._compute_kid, encoder_type='ae'),

        'morph': trainer._compute_morphological_features,

        'Geom dist SSL': partial(trainer._compute_geometric_distance, encoder_type='simclr'),
        'Geom dist AE': partial(trainer._compute_geometric_distance, encoder_type='ae'),

        'ACA': partial(trainer._attribute_control_accuracy, build_hist=False),
    }

    results = {}
    for name, method in eval_methods.items():
        for _ in trange(n, desc=name):
            val = method()

            if name == 'morph':
                for key, val in val.items():
                    curr_name = f'{name}_{key}'
                    if curr_name not in results:
                        results[curr_name] = []
                    results[curr_name].append(val)
            elif name == 'ACA':

                if name not in results:
                    results[name] = []

                results[name].append(val['aggregated_attribute_accuracy'])
            else:
                if name not in results:
                    results[name] = []
                results[name].append(val)

    for name, vals in results.items():
        print(f'Method: {name}. Mean: {np.mean(vals)}, STD: {np.std(vals)}')

    with open('./runs/results_morph.json', 'w') as f:
        json.dump(str(results), f)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--n', '-n', type=int, default=100)
    args = parser.parse_args()
    main(args)
