import argparse

from src.evaluation import SingleImageEvaluator


def run(config_path: str):
    evaluator = SingleImageEvaluator(config_path)
    evaluator.evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c',
                        type=str)
    args = parser.parse_args()
    run(args.config)

