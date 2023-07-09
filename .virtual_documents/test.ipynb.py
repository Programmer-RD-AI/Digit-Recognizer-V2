import torch
import torchvision
from torch import nn,optim
from torch.utils.data import Dataset,DataLoader
import wandb
import random
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
<<<<<<< Updated upstream
=======
import matplotlib.pyplot as plt
>>>>>>> Stashed changes
import threading
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'


<<<<<<< Updated upstream
=======
data = pd.read_csv('ML/data/train.csv')


data


plt.imshow(np.array(data.iloc[10].tolist()[1:]).reshape(28,28,1))


np.sqrt(784)














>>>>>>> Stashed changes

