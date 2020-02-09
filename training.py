import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision

from Net import Net
from helpers import get_train_set_and_loader, show_image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = Net()
net.to(device)

_, train_loader = get_train_set_and_loader()

loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    net.parameters(),
    lr=0.001,
    momentum=0.9
)

writer = SummaryWriter('runs/fashion_mnist_experiment_1')

dataiter = iter(train_loader)
images, labels = dataiter.__next__()

img_grid = torchvision.utils.make_grid(images)
show_image(img_grid, one_channel=True)

writer.add_image('four_fashion_mnist_images', img_grid)
writer.add_graph(net, images.to(device))
writer.close()
