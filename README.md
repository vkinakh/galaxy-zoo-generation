![Python 3.9](https://img.shields.io/badge/python-3.8-green.svg)

# Evaluation Metrics for Galaxy Image Generators
## Galaxy Zoo generation using information-theoretic stochastic contrastive conditional GAN: InfoSCC-GAN

This repo contains the code for the paper: "[Evaluation Metrics for Galaxy Image Generators](http://dx.doi.org/10.2139/ssrn.4276472)". Pytorch implementation of the Galaxy Zoo generation using InfoSCC-GAN framework.

Check out our [Hugginface Space](https://huggingface.co/spaces/vitaliykinakh/Galaxy_Zoo_Generation)


## Installation
### Conda installation
```bash
conda env create -f environment.yml
```

## Dataset
We use dataset from [Kaggle competition](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge).

## Training
### Training of the encoder
To run the training of the encoder (SimCLR) first fill the **config file**. Example of detailed config file is available `configs/galaxy_zoo_encoder.yaml`.

Then run
```bash
python main.py --mode train --task encoder --config <path to config>
```

### Training of the classifier
To run the training of the classifier, first fill the **config file**. Example of detailed config file is available: `galaxy_zoo_classification.yml`

Then run
```bash
python main.py --mode train --task classifier --config <path to config>
```

### Training of the generator
To run the training of the generator, fill the **config file**. Examples of detailed config is available: `configs/galaxy_zoo_generation.yml`.

Then run
```bash
python main.py --mode train --task generator --config <path to config>
```

## Evaluation
### Evaluation of the generator

Evaluation of the generator includes: 
- compute **FID** score with InceptionV3 encoder features
- compute **Inception Score (IS)**
- compute **Chamfer distance**
- compute **FID** score with SimCLR encoder features
- compute **FID** score with autoencoder (AE) features
- compute **perceptual path length (PPL)** with SimCLR encoder features
- compute **perceptual path length (PPL)** with VGG16 encoder features
- compute **perceptual path length (PPL)** with AE features
- compute **KID** with InceptionV3 encoder features
- compute **KID** with SimCLR encoder features
- compute **KID** with AE features 
- compute morphological features of the generated samples
- compute **geometrical distance** with SimCLR encoder features
- compute **geometrical distance** with AE features  
- perform attribute control accuracy 
- traverse z1, ... zk variables
- explore epsilon variable.

To run the evaluation, first fill the **config file**, put path to the generator in fine_tune_from field. Then run

```bash
python main.py --mode evaluate --task generator --config <path to config>
```

## Compute baseline metrics values

Computing baseline metrics values includes:
- compute **FID** score with InceptionV3 encoder between two splits of dataset
- compute **FID** score with SimCLR encoder between two splits of dataset
- compute **Chamfer distance** between two splits of dataset
- compute **KID** score with InceptionV3 encoder between two splits of dataset
- compute **KID** score SimCLR encoder between two splits of dataset
- compute **Geometric distance** between two splits of dataset.

To compute baseline metrics values, first fill the **config file**. The example is in `configs/galaxy_zoo_baseline.yml`. Then run
```bash
python compute_baseline.py --config <path to config>
```

## Simulate mode-collapsed model and evaluate it

To simulate mode-collapsed model (model that returns the same sample all the time) and evaluate it, fill the **config 
file**. The example is in `configs/single_image_eval.yml`. Then run

```bash
python evaluate_single_image_generator.py --config <path to config>
```

## Pretrained models
|Model type|Download link|
|----------|-------------|
|Encoder   |[Download](https://drive.google.com/file/d/1lOXiTBcbI3AnoNiFmrk_1keQVKqbAwjB/view?usp=sharing)|
|Classifier (for training of the generator)|[Download](https://drive.google.com/file/d/1B9SMUFFldvDEgHrUQVmFTPSxuiRZ3sfk/view?usp=sharing)|
|Autoencoder|[Download](https://drive.google.com/file/d/1WTj-x3LjbIufdypnr4GQD1bzYyyPPAY4/view?usp=sharing)|
|Classifier (for evaluation)|[Download](https://drive.google.com/file/d/1Ogjajeo5KH5mhaHseNsNVMfHfI3GG7Kd/view?usp=sharing)|


## Citation
```bash
@article{hackstein2023evaluation,
  title={Evaluation metrics for galaxy image generators},
  author={Hackstein, Stefan and Kinakh, Vitaliy and Bailer, Christian and Melchior, Martin},
  journal={Astronomy and Computing},
  pages={100685},
  year={2023},
  publisher={Elsevier}
}
```