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
import math

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["WANDB_SILENT"] = "true"
PROJECT_NAME = "Digit Recognizer V2"
# device = torch.cuda.set_device(0)
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)
from ML.metrics import *
from ML.helper import *
from ML.data_loader import *
from ML.modelling import *
