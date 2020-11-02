import pandas as pd
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.optim as optim
import torch
import torch.nn as nn
from torchsummary import summary

from network import Net
from helpers import make_batch, make_batch_classification, normalize, adjust_learning_rate, unnormalize_tensor

from oriented_iou_loss import cal_diou


def train(model, optimizer, epoch, device, steps, batch_size, criterion, classification=False):
    model.train()
    running_loss = 0.0
    running_iou = 0.0

    for _ in range(0, steps):
        if classification:
            images, target = make_batch_classification(batch_size)
        else:
            images, target = make_batch(batch_size)

        images = images.to(device)
        target = target.to(device)

        output = model(images)

        if classification:
            loss = criterion(output, target)
        else:
            loss, iou = criterion(
                torch.unsqueeze(unnormalize_tensor(output), 1),
                torch.unsqueeze(unnormalize_tensor(target), 1),
                'smallest'
            )
            loss = torch.mean(loss)
            iou = torch.mean(iou)
            running_iou += iou.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(epoch)
    print(running_loss / steps)

    if not classification:
        print(running_iou / steps)



def main():
    model = Net()

    # Part I - Train model to localize spaceship on images containing spaceship
    print("Start localization training")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), eps=1e-07)

    cudnn.benchmark = True
    criterion = cal_diou

    epochs = 40
    steps_per_epoch = 3125
    batch_size = 64

    for epoch in range(0, epochs):
        adjust_learning_rate(optimizer, epoch)
        train(model, optimizer, epoch, device, steps_per_epoch, batch_size, criterion)

    # Part II - Apply transfer learning to train pre-trained model to detect whether spaceship exists
    print("Start classification training")

    model.mode = 'classification'
    criterion = nn.BCELoss()

    for param in model.convnet.parameters():
        param.requires_grad = False

    for param in model.localizer.parameters():
        param.requires_grad = False

    batch_size = 64
    steps_per_epoch = 500
    epochs = 2

    optimizer = optim.Adam(model.parameters(), eps=1e-07)

    for epoch in range(epochs):
        train(model, optimizer, epoch, device, steps_per_epoch, batch_size, criterion, classification=True)

    # Save model
    path = F'model.pth.tar'
    torch.save(model.state_dict(), path)


if __name__ == "__main__":
    main()
