import torch

from src.models import ImageClassifier


if __name__ == '__main__':

    classifier = ImageClassifier()
    path = './models/classifier/parameter_state_dict_ImageClassifier.pth'

    ckpt = torch.load(path)
    res = classifier.load_state_dict(ckpt)
    print(res)
