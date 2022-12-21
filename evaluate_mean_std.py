from argparse import ArgumentParser
from functools import partial
import json
from collections import namedtuple
from pprint import pprint

from tqdm import trange
import numpy as np

from src.trainer import GalaxyZooInfoSCC_Trainer
from src.utils import get_config


MeanStddev = namedtuple("Mean", "Mean Stddev")


def get_mean_dicts(dicts, keys=None):
    if keys is None:
        keys = []
    values = np.array([[d[k] for d in dicts] for k in keys])
    return {k: MeanStddev(v, s) for k, v, s in zip(keys, values.mean(axis=1), values.std(axis=1))}


def get_mean_nested_dicts(dicts, keys=None):
    if keys is None:
        keys = []
    results = {}
    for k in keys:
        sub_dicts = [d[k] for d in dicts]
        result = get_mean_dicts(sub_dicts, keys=sub_dicts[0].keys())
        results[k] = result
    return results


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

        'cluster': trainer._compute_distribution_measures,
    }

    results = {}
    results_clusters = []
    results_wasserstein = []

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
            elif name == 'cluster':
                results_clusters.append(val['cluster'])

                res_wasserstein = val['wasserstein']
                results_wasserstein.append(res_wasserstein)
            else:
                if name not in results:
                    results[name] = []
                results[name].append(val)

    for name, vals in results.items():
        print(f'Method: {name}. Mean: {np.mean(vals)}, STD: {np.std(vals)}')

    result_clusters = get_mean_nested_dicts(results_clusters, keys=['distances', 'errors'])
    result_wasserstein = get_mean_dicts(results_wasserstein, keys=res_wasserstein.keys())
    print('clusters')
    pprint(result_clusters)
    print('Wasserstein')
    pprint(result_wasserstein)

    with open('./runs/results_mean_std_13_clusters.json', 'w') as f:
        json.dump(str(results), f)
        json.dump(str(result_clusters), f)
        json.dump(str(result_wasserstein), f)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--n', '-n', type=int, default=100)
    args = parser.parse_args()
    main(args)
