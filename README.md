# DigiVis: Neural Network Vision for Digit Classification

A deep learning implementation for MNIST digit recognition using convolutional neural networks and computer vision techniques. This project combines modern neural architectures with advanced image processing for accurate digit classification.

## Overview

DigiVis is a comprehensive implementation of various neural network architectures for digit recognition, utilizing the MNIST dataset. The project incorporates modern deep learning practices including data normalization, image transformations, and model evaluation metrics.

## Features

- Multiple neural network architectures (CNN and Linear models)
- Data normalization and preprocessing
- Image transformations and augmentations
- Model training with performance metrics
- Weights & Biases integration for experiment tracking
- Comprehensive test suite
- CUDA support for GPU acceleration

## Requirements

- Python 3.x
- PyTorch
- torchvision
- numpy
- pandas
- Pillow
- wandb
- matplotlib
- scikit-learn
- tqdm

## Installation

1. Clone the repository

```bash
git clone https://github.com/Programmer-RD-AI/DigiVis.git
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

Run the main training script:

```
python run.py
```

For interactive exploration, use the provided Jupyter notebook:

```
jupyter notebook test.ipynb
```

## Model Configuration

- Image Size: 224x224
- Batch Size: 32
- CUDA enabled for GPU acceleration
- Random seed: 42 for reproducibility

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
