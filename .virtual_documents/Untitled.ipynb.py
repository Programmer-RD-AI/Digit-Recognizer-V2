import os
import random
import threading
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchinfo
import torchvision
import wandb
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import *
from tqdm import tqdm
from wandb import AlertLevel

PROJECT_NAME = "Digit Recognizer V2"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)

n = Normalizer(path="ML/data/train.csv", label_col="label")
n.mean(), n.std(), n.create_long_list().shape

data = pd.read_csv("ML/data/train.csv").drop("label", axis=1)

tot = 0
for i in tqdm(range(len(self.data))):
    for x in range(len(self.data.iloc[i].tolist())):
        tot += self.data.iloc[i].tolist()[x] / 255

n.mean()
