import torch
import torch.nn as nn
import torchvision
import numpy as np
import os
# from torchsummary import summary
import random
from skimage.color import rgb2lab, lab2rgb
from torch.utils.data import Dataset
import torch.nn.functional as F

from torchvision.io import read_image

home = '../Image colorization/dataset/landscape Images/'
total_images = len(os.listdir(home + 'color'))


class DataEncoder(Dataset):
    def __init__(self, indices, img_dir, transform):
        self.img_dim = img_dir
        self.transform = transform
        self.img_indices = indices
        self.gray_path = img_dir + 'gray/'
        self.color_path = img_dir + 'color/'

    def __len__(self):
        return len(self.img_indices)

    def __getitem__(self, idx):
        img_name = str(idx) + '.jpg'
        image = read_image(self.gray_path + img_name)
        image = image.unsqueeze(0)
        image = F.interpolate(image, (160, 160))
        image = image.squeeze(0)
        image = image.permute(1, 2, 0)
        image = image.repeat(1, 1, 3)
        image = image.permute(2, 0, 1)
        label = read_image(self.color_path + img_name)
        label = label.unsqueeze(0)
        label = F.interpolate(label, (160, 160))
        label = label.squeeze(0)
        label = label.permute(1, 2, 0)
        label = label.permute(2, 0, 1)
        image = torch.tensor(rgb2lab(image.permute(1, 2, 0) / 255))
        label = torch.tensor(rgb2lab(label.permute(1, 2, 0) / 255))

        image = (image + torch.tensor([0, 128, 128])) / torch.tensor([100, 255, 255])
        label = (label + torch.tensor([0, 128, 128])) / torch.tensor([100, 255, 255])

        image = image.permute(2, 0, 1)
        label = label.permute(2, 0, 1)

        image = image[:1, :, :]
        label = label[1:, :, :]
        return image, label


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        # self.pool = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(2,2)

        # Decoder Layer
        self.t_conv0 = nn.ConvTranspose2d(512, 256, 3,stride=2, padding=1, output_padding=1)
        self.t_conv1 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.t_conv2 = nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1)
        self.t_conv3 = nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1)
        self.t_conv4 = nn.ConvTranspose2d(192, 15, 3, stride=1, padding=1)
        self.dropout = nn.Dropout(0.2)
        self.converge = nn.Conv2d(16, 2, 3, stride=1, padding=1)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        x5 = F.relu(self.conv5(x4))
        xd = F.relu(self.t_conv0(x5))
        xd = F.relu(self.t_conv1(x4))
        xd = torch.cat((xd, x3), dim=1)
        xd = self.dropout(xd)
        xd = F.relu(self.t_conv2(xd))
        xd = torch.cat((xd, x2), dim=1)
        xd = self.dropout(xd)
        xd = F.relu(self.t_conv3(xd))
        xd = torch.cat((xd, x1), dim=1)
        xd = self.dropout(xd)
        xd = F.relu(self.t_conv4(xd))
        xd = torch.cat((xd, x), dim=1)
        x_out = F.relu(self.converge(xd))
        return x_out




model = AutoEncoder()
# print(summary(model, input_size=(1, 160, 160)))
