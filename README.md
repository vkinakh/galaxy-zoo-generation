# Galaxy Zoo generation using information-theoretic stochastic contrastive conditional GAN: InfoSCC-GAN

This repo contains Pytorch implementation of the Galaxy Zoo generation using InfoSCC-GAN framework.

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
- compute **FID** score with InceptionV3 encoder
- compute **Inception Score (IS)**
- compute **Chamfer distance**
- compute **FID** score with SimCLR encoder 
- compute **perceptual path length (PPL)** with SimCLR encoder
- compute **perceptual path length (PPL)** with VGG16 encoder
- compute **KID** with InceptionV3 encoder
- compute **KID** with SimCLR encoder
- compute morphological features of the generated samples
- compute **geometrical distance**
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

## Pretrained models
|Model type|Download link|
|----------|-------------|
|Encoder   |[Download](https://drive.google.com/file/d/1lOXiTBcbI3AnoNiFmrk_1keQVKqbAwjB/view?usp=sharing)|
|Classifier|[Download](https://drive.google.com/file/d/1B9SMUFFldvDEgHrUQVmFTPSxuiRZ3sfk/view?usp=sharing)|
