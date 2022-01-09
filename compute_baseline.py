import argparse

from src.trainer import GalaxyZooInfoSCC_Trainer
from src.utils import get_config


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c',
                        type=str,
                        required=True,
                        help='Path to config for baseline computation')

    args = parser.parse_args()
    config_path = args.config

    config = get_config(config_path)
    trainer = GalaxyZooInfoSCC_Trainer(config_path, config)
    trainer.compute_baseline()
