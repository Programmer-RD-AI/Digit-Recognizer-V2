from torchvision import *
import torchvision, torch
from torch import nn, optim

model = torchvision.models.vgg11()
model.features[0] = nn.Conv2d(
    1,
    64,
    kernel_size=3,
    stride=1,
    padding=1,
)
model.classifier[6] = nn.Linear(4096, 10)

print(model)
