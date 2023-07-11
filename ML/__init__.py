import math
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

from ML.data_loader import *
from ML.helper import *
from ML.metrics import *
from ML.modelling import *

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["WANDB_SILENT"] = "true"
PROJECT_NAME = "Digit Recognizer V2"
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)
