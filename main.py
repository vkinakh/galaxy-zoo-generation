import argparse

from src.trainer import GalaxyZooSimCLRTrainer
from src.trainer import GalaxyZooClassifierTrainer
from src.trainer import GalaxyZooConditionalGeneratorTrainer
from src.trainer import GalaxyZooInfoSCC_Trainer
from src.utils import PathOrStr
from src.utils import get_config


def train_encoder(config_path: PathOrStr):
    config = get_config(config_path)
    trainer = GalaxyZooSimCLRTrainer(config_path, config)
    trainer.train()


def train_classifier(config_path: PathOrStr):
    config = get_config(config_path)
    trainer = GalaxyZooClassifierTrainer(config_path, config)
    trainer.train()


def train_generator(config_path: PathOrStr):
    config = get_config(config_path)
    # trainer = GalaxyZooConditionalGeneratorTrainer(config_path, config)
    trainer = GalaxyZooInfoSCC_Trainer(config_path, config)
    trainer.train()


def evaluate_generator(config_path: PathOrStr):
    config = get_config(config_path)
    trainer = GalaxyZooConditionalGeneratorTrainer(config_path, config)
    trainer.evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m',
                        type=str,
                        choices=['train', 'evaluate'])
    parser.add_argument('--task', '-t',
                        type=str,
                        choices=['encoder', 'classifier', 'generator'])
    parser.add_argument('--config', '-c',
                        type=str)
    args = parser.parse_args()

    config = args.config
    mode = args.mode
    task = args.task

    if mode == 'train':

        if task == 'encoder':
            train_encoder(config)
        elif task == 'classifier':
            train_classifier(config)
        elif task == 'generator':
            train_generator(config)

    elif mode == 'evaluate':

        if task == 'generator':
            evaluate_generator(config)
