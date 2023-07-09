import torch
import torchvision
import os
import random
import wandb
from torchvision.models import *
from torch import nn, optim
import threading
import torchinfo
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from typing import *
from sklearn.model_selection import train_test_split
from torchvision import transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wandb import AlertLevel

PROJECT_NAME = "Digit Recognizer V2"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)





n=Normalizer(path="ML/data/train.csv",label_col='label')
n.mean(),n.std(),n.create_long_list().shape


data = pd.read_csv("ML/data/train.csv").drop('label', axis=1)


tot = 0
for i in tqdm(range(len(self.data))):
    for x in range(len(self.data.iloc[i].tolist())):
        tot += self.data.iloc[i].tolist()[x] / 255



n.mean()



