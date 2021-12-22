from src.trainer import GalaxyZooInfoSCC_Trainer
from src.utils import get_config


if __name__ == '__main__':

    config_path = '/home/kinakh/Development/Projects/galaxy_zoo_generation/configs/galaxy_zoo_baseline.yml'
    config = get_config(config_path)

    trainer = GalaxyZooInfoSCC_Trainer(config_path, config)
    trainer.compute_baseline()
