import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from Net import Net
from classes import classes
from helpers import get_train_set_and_loader, select_n_random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = Net()
net.to(device)

train_set, train_loader = get_train_set_and_loader()

loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    net.parameters(),
    lr=0.001,
    momentum=0.9
)

writer = SummaryWriter('runs/fashion_mnist_experiment_1')

dataiter = iter(train_loader)
images, labels = select_n_random(train_set.data, train_set.targets)

class_labels = [classes[label] for label in labels]

features = images.view(-1, 28 * 28)
writer.add_embedding(
    features,
    metadata=class_labels,
    label_img=images.unsqueeze(1)
)
writer.close()
