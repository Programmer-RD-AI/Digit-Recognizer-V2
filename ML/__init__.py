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

from ML.data_loader import *
from ML.helper_funcs import *
from ML.metrics import *
from ML.modelling import *

PROJECT_NAME = "Digit Recognizer V2"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)
