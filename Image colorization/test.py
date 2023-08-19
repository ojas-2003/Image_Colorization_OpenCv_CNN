import torch
from matplotlib import pyplot as plt
from skimage.color import lab2rgb
from torch.utils.data import DataLoader
import os
import random

from torchvision.io import read_image
import torch.nn.functional as F

from model import AutoEncoder, DataEncoder
import torch.nn as nn
import skimage
import torch.optim as optim
from tqdm import tqdm

home = '../Image colorization/dataset/landscape Images/'
total_img = len(os.listdir(home + 'color'))
print(total_img)
rand_ind = random.sample(list(range(total_img)), total_img)
sample_size = round(total_img*0.8)
train_sample = rand_ind[:sample_size]
test_sample = rand_ind[sample_size:]

# print(len(train_sample), " ", len(test_sample))

import torchvision.transforms as transforms

train_trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    ]

)

test_trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    ]
)

train_dataset = DataEncoder(indices= train_sample, img_dir=home, transform = train_trans)
test_dataset = DataEncoder(indices= test_sample, img_dir=home, transform = test_trans)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

img,label = next(iter(train_loader))
sample_image, sample_label = img[0], label[0]
print(sample_label.shape, sample_image.shape)


model = AutoEncoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epoch = 1
train_losses = []
test_losses = []



model = AutoEncoder()
model.load_state_dict(torch.load('col.pth', map_location=torch.device('cpu')))

i=20
while i<30:
    test_img,test_label = next(iter(train_loader))
    pred = model.forward(test_img[0].float().view(1,1,160,160))
    lab_pred = torch.cat((test_img[0].view(1,160,160),pred[0].cpu()),dim=0)
    lab_pred_inv_scaled = lab_pred.permute(1,2,0) * torch.tensor([100,255,255]) - torch.tensor([0,128,128])
    rgb_pred = lab2rgb(lab_pred_inv_scaled.detach().numpy())
    fig = plt.figure(figsize=(10,10))
    plt.subplot(221)
    plt.imshow(test_img[0].permute(1,2,0),cmap='gray')
    plt.title('GrayScale Image')
    plt.subplot(222)
    plt.imshow(rgb_pred)
    plt.title('Predicted Color Image')
    plt.show()
    i+=1

