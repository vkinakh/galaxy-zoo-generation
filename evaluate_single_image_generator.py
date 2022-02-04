from src.evaluation import SingleImageEvaluator


def run(config_path: str):
    evaluator = SingleImageEvaluator(config_path)
    evaluator.evaluate()


if __name__ == '__main__':
    run('./configs/single_image_eval.yml')
