from collections import Counter
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.nn import MSELoss
from torch.nn import Sequential, Dropout, Flatten
from torch.optim import Adam
import torchvision.models as models
from torchvision.transforms import Compose, FiveCrop, Lambda
from torchvision.transforms.functional import rotate, hflip
from torch.optim.lr_scheduler import MultiStepLR

from .galaxy_zoo_classifier import MaxOut, ALReLU
from src.utils.labeling import class_groups_indices, make_galaxy_labels_hierarchical
from src.metrics.distribution_measures.neural_network import NeuralNetwork


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class_group_layers = {
    1: [1, 6],
    2: [2, 7],
    3: [3, 9],
    4: [4, 5],
    5: [8, 10, 11],
}

mse = MSELoss()


def measure_accuracy_classifier_group(labels_prediction: torch.Tensor,
                                      labels_target: torch.Tensor,
                                      minimum_answers=0.5,
                                      allowed_deviation=0.1
                                      ) -> float:
    """
    measure accuracy of classifier prediction for individual group.
    for questions answered by at least miminum_answers of people
    return relative amount of agreements for maxed label

    Parameters
    ----------
    labels_prediction(torch.tensor) : labels predicted by classifier
    labels_target(torch.tensor) : original labels
    minimum_answer(float) : (0.0 to 1.0) minimum amount of collected answers on a question to be considered
    """
    consider = torch.sum(labels_target, dim=1) > minimum_answers
    pred = torch.argmax(labels_prediction[consider], dim=1)
    targ = torch.argmax(labels_target[consider], dim=1)
    accuracy = torch.mean((pred == targ).float()).item()
    return accuracy


def measure_accuracy_classifier(labels_prediction: torch.Tensor,
                                labels_target: torch.Tensor,
                                minimum_answers=0.5,
                                allowed_deviation=0.1, considered_groups=range(1, 12)
                                ) -> dict:
    """
    measure accuracy of classifier prediction. Calculates accuracy on label groups
    where at least miminum_answers of people answered this question
    returns loss for each group

    Parameters
    ----------
    labels_prediction(torch.tensor) : labels predicted by classifier
    labels_target(torch.tensor) : original labels
    minimum_answer(float) : (0.0 to 1.0) minimum amount of collected answers on a question to be considered
    """
    measure_accuracy = partial(measure_accuracy_classifier_group,
                               minimum_answers=minimum_answers,
                               allowed_deviation=allowed_deviation)
    accuracies = {group: measure_accuracy(labels_prediction[:, ix], labels_target[:, ix])
                  for group, ix in class_groups_indices.items()
                  if group in considered_groups}
    return accuracies


def get_sample_variance(sample: torch.Tensor) -> torch.Tensor:
    """ calculate variance of sample via MSE from mean"""
    avg = torch.mean(sample, dim=0, keepdims=True)
    sample_variance = mse(sample, avg.repeat(len(sample), *(1,)*(sample.dim()-1)))
    return sample_variance


def loss_sample_variance(features: torch.Tensor, threshold=0.001) -> torch.Tensor:
    """ calculate loss of sample variance as mean squared error from average """
    sample_variance = get_sample_variance(features)
    loss = torch.max(torch.tensor(0, device=device), threshold - sample_variance)
    return loss


class Losses:
    """ Container for all kinds of losses of Neural Networks

        Parameter
        ---------
        type: str, loss category used as plot title
        label: str, identifier, used as plot label
        log: bool, indicate whether y-axis is log scaled in plot
        rate: bool, indicate whether to space y from 0 to 1

        Usage
        -----
    """

    def __init__(self, type: str, label: str, log=True, rate=False):
        self.label = label
        self.type = type
        self.losses = {}

    def append(self, iteration: int, loss: torch.Tensor) -> None:
        if loss is torch.Tensor:
            loss = loss.item()
        self.losses[iteration] = loss

    def plot(self, log=True, **kwargs):
        try:
            x, y = zip(*self.losses.items())
        except:
            return
        plt.plot(x, y, label=self.label, **kwargs)
        plt.xlabel("iteration")
        plt.title(self.type)
        plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
        if log:
            plt.yscale("log")
        plt.grid(True)
        #        plt.yaxis.set_ticks_position("both")
        plt.tight_layout()


class Accuracies(Losses):
    """ Container for accuracy measures """

    def plot(self, **kwargs):
        super(Accuracies, self).plot(log=False)
        plt.ylim(-0.1, 1.1)


class ConsiderGroups:
    def __init__(self,
                 considered_groups: list = list(range(12)),  ## groups to be considered from start
                 ):
        self.considered_groups = []
        self.considered_label_indices = []
        for group in considered_groups:
            self.consider_group(group)

    def consider_group(self, group: int) -> None:
        """ add group to considered_label_indices """
        if group in self.considered_groups:
            print(f"group {group} already considered")
            return;
        self.considered_groups.append(group)
        self.considered_label_indices.extend(class_groups_indices[group])
        self.considered_label_indices.sort()

    def unconsider_group(self, group: int) -> None:
        """ add group to considered_label_indices """
        if group not in self.considered_groups:
            print(f"group {group} not considered")
            return;
        self.considered_groups.remove(group)
        for label in class_groups_indices[group]:
            self.considered_label_indices.remove(label)
        self.considered_label_indices.sort()

    def get_considered_labels(self) -> list:
        """ returns list of considered label indices """
        return self.considered_label_indices

    def get_labels_dim(self):
        """ obtain dimensions of label vector for considered groups """
        return len(self.considered_label_indices)


class ClassifierBase(NeuralNetwork):
    def __init__(self,
                 considered_groups: list = range(1, 8),  # groups of labels to be considered
                 ):
        super(ClassifierBase, self).__init__()
        self.considered_groups = ConsiderGroups(considered_groups)
        self.considered_label_indices = self.considered_groups.get_considered_labels()

    def consider_groups(self, *groups: int) -> None:
        """ add group to considered_groups """
        for group in groups:
            self.considered_groups.consider_group(group)
        self.considered_label_indices = self.considered_groups.get_considered_labels()

    def consider_layer(self, layer: int) -> None:
        """ add groups in layer to considered_groups """
        self.consider_groups(*class_group_layers[layer])


class ImageClassifier(ClassifierBase):
    """ model for morphological classification of galaxy images
    Usage
    -----
    to use pretrained model, do
    >>> classifier = ImageClassifier()
    >>> classifier.load()
    >>> classifier.eval()
    >>> classifier.use_label_hierarchy()
    >>> labels = classifier(images)
    """

    def __init__(self,
                 seed=None,
                 optimizer=Adam, optimizer_kwargs={},
                 learning_rate_init=0.04,
                 gamma=0.995,  # learning rate decay factor
                 considered_groups=list(range(12)),  ## group layers to be considered from start
                 sample_variance_threshold=0.002,
                 weight_loss_sample_variance=0,  # 10.
                 evaluation_steps=250,  # number of batches between loss tracking
                 N_batches_test=1,  # number of batches considered for evaluation
                 ):
        super(ImageClassifier, self).__init__(considered_groups=considered_groups)
        if seed is not None:
            torch.manual_seed(seed)

        # '''
        resnet = models.resnet18(pretrained=False)
        self.conv = Sequential(
            *(list(resnet.children())[:-1]),
            Flatten(),
        )
        '''  architecture used by Dielemann et al 2015
        self.conv = Sequential(
#            Conv2dUntiedBias(41, 41, 3, 32, kernel_size=6),
            Conv2d(3,32, kernel_size=6),
            ReLU(),
            MaxPool2d(2),
#            Conv2dUntiedBias(16, 16, 32, 64, kernel_size=5),
            Conv2d(32, 64, kernel_size=5),
            ReLU(),
            MaxPool2d(2),
#            Conv2dUntiedBias(6, 6, 64, 128, kernel_size=3),
            Conv2d(64, 128, kernel_size=3),
            ReLU(),
#            Conv2dUntiedBias(4, 4, 128, 128, kernel_size=3), #weight_std=0.1),
            Conv2d(128, 128, kernel_size=3),
            ReLU(),
            MaxPool2d(2),
            Flatten(),
        )
        #'''
        self.dense1 = MaxOut(8192, 2048, bias=0.01)
        self.dense2 = MaxOut(2048, 2048, bias=0.01)
        self.dense3 = Sequential(
            MaxOut(2048, 37, bias=0.1),
            #            LeakyReLU(negative_slope=1e-7),
            ALReLU(negative_slope=1e-2),
        )
        self.dropout = Dropout(p=0.5)

        self.augment = Compose([
            Lambda(lambda img: torch.cat([img, hflip(img)], 0)),
            Lambda(lambda img: torch.cat([img, rotate(img, 45)], 0)),
            FiveCrop(45),
            Lambda(lambda crops: torch.cat([rotate(crop, ang) for crop, ang in zip(crops, (0, 90, 270, 180))], 0)),
        ])
        self.N_augmentations = 16
        self.N_conv_outputs = 512

        self.set_optimizer(optimizer, lr=learning_rate_init, **optimizer_kwargs)
        #        self.scheduler = ExponentialLR(self.optimizer, gamma=gamma)
        self.scheduler = MultiStepLR(self.optimizer, milestones=[292, 373], gamma=gamma)

        # if True, output probabilities are renormalized to fit the hierarchical label structure
        self.make_labels_hierarchical = False
        self.N_batches_test = N_batches_test
        self.evaluation_steps = evaluation_steps  # number of batches between loss tracking
        self.weight_loss_sample_variance = weight_loss_sample_variance
        self.sample_variance_threshold = sample_variance_threshold

        self.iteration = 0
        self.epoch = 0
        self.losses_train = Losses("loss", "train")
        self.losses_valid = Losses("loss", "valid")
        self.sample_variances_train = Losses("sample variance", "train")
        self.sample_variances_valid = Losses("sample variance", "valid")
        for g in range(1, 12):
            setattr(self, f"accuracies_Q{g}_train", Accuracies("accuracy train", f"Q{g}"))
            setattr(self, f"accuracies_Q{g}_valid", Accuracies("accuracy valid", f"Q{g}"))
        self.losses_regression = Losses("loss", "regression")
        self.losses_variance = Losses("loss", "sample variance")

        # return to random seed
        if seed is not None:
            sd = np.random.random() * 10000
            torch.manual_seed(sd)

    def update_optimizer_learningrate(self, learning_rate) -> None:
        print("update lr", learning_rate)
        for i in range(len(self.optimizer.param_groups)):
            self.optimizer.param_groups[i]['lr'] = learning_rate

    def use_label_hierarchy(self) -> None:
        self.make_labels_hierarchical = True

    def forward(self, x: torch.Tensor, train=False) -> torch.Tensor:
        x = self.augment(x)
        x = self.conv(x)

        x = self.recombine_augmentation(x)

        x = self.dropout(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)
        x = self.dense3(x)
        #        x += 1e-4  ## use only with LeakyReLU to prevent values < 0
        if self.make_labels_hierarchical:
            x = make_galaxy_labels_hierarchical(x)
        return x

    def recombine_augmentation(self, x) -> torch.Tensor:
        """ recombine results of augmented views to single vector """
        batch_size = x.size(0) // self.N_augmentations
        x = x.reshape(self.N_augmentations, batch_size, self.N_conv_outputs)
        x = x.permute(1, 0, 2)
        x = x.reshape(batch_size, self.N_augmentations * self.N_conv_outputs)
        return x

    def train_step(self, images: torch.tensor, labels: torch.tensor) -> float:
        self.train()
        labels_pred = self.forward(images, train=True)
        loss_regression = mse(labels_pred[:, self.considered_label_indices], labels[:, self.considered_label_indices])
        loss_variance = self.weight_loss_sample_variance * \
                        loss_sample_variance(labels_pred[:, self.considered_label_indices],
                                             threshold=self.sample_variance_threshold)
        loss = loss_regression + loss_variance
        self.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)
        self.optimizer.step()
        self.iteration += 1
        return loss.item()

    def train_epoch(self,
                    data_loader_train: torch.utils.data.DataLoader,
                    data_loader_valid: torch.utils.data.DataLoader,
                    track: bool = False,
                    ) -> None:
        for images, labels in tqdm(data_loader_train, desc=f"epoch {self.epoch}"):
            images = images.to(device)
            labels = labels.to(device)
            loss = self.train_step(images, labels)
            if np.isnan(loss):
                from pdb import set_trace
                set_trace()
                loss = self.train_step(images, labels)
                raise Exception("loss is NaN")
            if not self.iteration % self.evaluation_steps - 1:
                loss_regression_train, loss_variance_train, accs_train, variance_train = \
                    self.evaluate_batch(images, labels, print_labels=False)
                loss_train = loss_regression_train + loss_variance_train * self.weight_loss_sample_variance
                self.losses_regression.append(self.iteration, loss_regression_train)
                self.losses_variance.append(self.iteration, loss_variance_train)
                self.losses_train.append(self.iteration, loss_train)
                self.sample_variances_train.append(self.iteration, variance_train)
                for group, acc in accs_train.items():
                    getattr(self, f"accuracies_Q{group}_train").append(self.iteration, acc)
                for images, labels in data_loader_valid:
                    images = images.to(device)
                    labels = labels.to(device)
                    break
                loss_regression_valid, loss_variance_valid, accs_valid, variance_valid = self.evaluate_batch(images,
                                                                                                             labels)
                loss_valid = loss_regression_valid + loss_variance_valid * self.weight_loss_sample_variance
                self.losses_valid.append(self.iteration, loss_valid)
                self.sample_variances_valid.append(self.iteration, variance_valid)
                for group, acc in accs_valid.items():
                    getattr(self, f"accuracies_Q{group}_valid").append(self.iteration, acc)
                if track:
                    import wandb
                    logs = {
                        "loss_regression_train": loss_regression_train,
                        "loss_variance_train": loss_variance_train,
                        "loss_train": loss_train,
                        "variance_train": variance_train,
                        "loss_regression_valid": loss_regression_valid,
                        "loss_variance_valid": loss_variance_valid,
                        "loss_valid": loss_valid,
                        "variance_valid": variance_valid,
                    }
                    logs.update({f"accuracy_Q{group}_train": acc for group, acc in accs_train.items()})
                    logs.update({f"accuracy_Q{group}_valid": acc for group, acc in accs_valid.items()})
                    wandb.log(logs)

        self.epoch += 1
        self.scheduler.step()
        self.save()

    def predict(self, images: torch.tensor) -> torch.Tensor:
        self.eval()
        return self(images)

    def evaluate_batches(self, data_loader: torch.utils.data.DataLoader) -> list:
        with torch.no_grad():
            loss = 0
            accs = Counter({group: 0 for group in range(1, 12)})
            variance = 0
            for N_test, (images, labels) in enumerate(data_loader):
                images = images.to(device)
                labels = labels.to(device)
                if N_test >= self.N_batches_test:
                    break
                loss_, accs_, variance_ = self.evaluate_batch(images, labels)
                loss += loss_
                accs.update(accs_)
                variance += variance_
            loss /= N_test + 1
            variance /= N_test + 1
            for group in accs.keys():
                accs[group] /= N_test + 1
        return loss, accs, variance

    def evaluate_batch(self, images: torch.tensor, labels: torch.tensor, print_labels=False) -> tuple:
        """ evaluations for batch """
        self.eval()
        with torch.no_grad():
            labels_pred = self.forward(images)
            if print_labels:
                for i, (prediction, target) in enumerate(zip(labels_pred, labels)):
                    print("target\t\t", np.around(target[self.considered_label_indices].cpu(), 3))
                    print("\033[1mprediction\t", np.around(prediction[self.considered_label_indices].cpu(), 3),
                          end="\033[0m\n")
                    if i >= 2:
                        break
                print("<target>\t", np.around(torch.mean(labels[:, self.considered_label_indices], dim=0).cpu(), 3))
                print("<target>\t", np.around(torch.std(labels[:, self.considered_label_indices], dim=0).cpu(), 3))
                print("\033[1m<prediction>\t",
                      np.around(torch.mean(labels_pred[:, self.considered_label_indices], dim=0).cpu(), 3),
                      end="\033[0m\n")
                print("\033[1m<prediction>\t",
                      np.around(torch.std(labels_pred[:, self.considered_label_indices], dim=0).cpu(), 3),
                      end="\033[0m\n")
            loss_regression = torch.sqrt(mse(labels_pred[:, self.considered_label_indices],
                                             labels[:, self.considered_label_indices]
                                             )).item()
            loss_variance = self.weight_loss_sample_variance * \
                            loss_sample_variance(labels_pred[:, self.considered_label_indices],
                                                 threshold=self.sample_variance_threshold
                                                 ).item()
            accs = measure_accuracy_classifier(labels_pred, labels,
                                               considered_groups=self.considered_groups.considered_groups)
            variance = get_sample_variance(labels_pred[:, self.considered_label_indices]).item()
        return loss_regression, loss_variance, accs, variance

    def plot_losses(self, save=False):
        self.losses_train.plot()
        self.losses_valid.plot()
        self.losses_regression.plot(linestyle=":")
        self.losses_variance.plot(linestyle=":")
        if save:
            plt.savefig("loss.png")
            plt.close()
        else:
            plt.show()

    def plot_sample_variances(self, save=False):
        self.sample_variances_train.plot()
        self.sample_variances_valid.plot()
        if save:
            plt.savefig("variances.png")
            plt.close()
        else:
            plt.show()

    def plot_accuracy(self, save=False):
        for group in range(1, 12):
            if not group in self.considered_groups.considered_groups:
                continue
            getattr(self, f"accuracies_Q{group}_train").plot()
        if save:
            plt.savefig("accuracy_train.png")
            plt.close()
        else:
            plt.show()

    def plot_test_accuracy(self, save=False):
        for group in range(1, 12):
            if not group in self.considered_groups.considered_groups:
                continue
            getattr(self, f"accuracies_Q{group}_valid").plot()
        if save:
            plt.savefig("accuracy_valid.png")
            plt.close()
        else:
            plt.show()
