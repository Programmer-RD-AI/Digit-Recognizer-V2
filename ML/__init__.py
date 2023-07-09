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

PROJECT_NAME = "Digit Recognizer V2"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)
from ML.metrics import *
from ML.helper_funcs import *
from ML.data_loader import *
from ML.modelling import *
