import torch
from torch.utils.data import DataLoader
import os
import random
# from model import AutoEncoder, DataEncoder
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

home = '../Image colorization/dataset/landscape Images/'
total_img = len(os.listdir(home + 'color'))
print(total_img)
rand_ind = random.sample(list(range(total_img)), total_img)
sample_size = round(total_img * 0.8)
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

train_dataset = DataEncoder(indices=train_sample, img_dir=home, transform=train_trans)
test_dataset = DataEncoder(indices=test_sample, img_dir=home, transform=test_trans)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

img, label = next(iter(train_loader))
sample_image, sample_label = img[0], label[0]
print(sample_label.shape, sample_image.shape)

model = AutoEncoder()
model = model.to('cuda')
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epoch = 40
train_losses = []
test_losses = []

for epoch in range(1, num_epoch + 1):
    train_loss = 0.0
    for data in tqdm(train_loader):
        images, labels = data
        images = images.float().to('cuda')
        labels = labels.float().to('cuda')
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)

    train_loss = train_loss / len(train_loader)
    train_losses.append(train_loss)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch,
        train_loss
    ))

    test_loss = 0

    with torch.no_grad():
        model.eval()
        for images, label in test_loader:
            images, label = images.to('cuda'), label.to('cuda')
            output = model(images)
            loss = criterion(output, label)
    model.train()

