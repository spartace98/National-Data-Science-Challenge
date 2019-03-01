import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 

from data import TrainingData


validation_percent = 0.1

# LOAD TRAINING DATA
td = TrainingData(validation_percent)
x_train = td.x_train
y_train = td.y_train
x_val = td.x_val
y_val = td.y_val


