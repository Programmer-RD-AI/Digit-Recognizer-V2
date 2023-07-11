from torchvision import *
import torchvision, torch
from torch import nn, optim

model = torchvision.models.resnet101()
# model.features[0][0] = nn.Conv2d(
#     1,
#     32,
#     kernel_size=3,
#     stride=2,
#     padding=1,
# bias=False
# )
# model.classifier[1] = nn.Linear(1280, 10)

print(model)
