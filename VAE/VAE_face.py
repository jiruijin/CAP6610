import glob as glob
import os
from tqdm import tqdm
def prepare_dataset(ROOT_PATH):
    image_dirs = os.listdir(ROOT_PATH)
    image_dirs.sort()
    print(len(image_dirs))
    print(image_dirs[:5])
    all_image_paths = []
    for i in tqdm(range(len(image_dirs))):
        image_paths = glob.glob(f"{ROOT_PATH}/{image_dirs[i]}/*")
        image_paths.sort()
        for image_path in image_paths:
            all_image_paths.append(image_path)
        
    print(f"Total number of face images: {len(all_image_paths)}")
    train_data = all_image_paths[:-2000]
    valid_data = all_image_paths[-2000:]
    print(f"Total number of training image: {len(train_data)}")
    print(f"Total number of validation image: {len(valid_data)}")
    return train_data, valid_data

import cv2
from torch.utils.data import Dataset
class LFWDataset(Dataset):
    def __init__(self, data_list, transform):
        self.data = data_list
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image = cv2.imread(self.data[index])
        image = cv2.resize(image, (64, 64))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        return image
    
import imageio
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image

to_pil_image = transforms.ToPILImage()
def transform():
    transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    return transform

def image_to_vid(images):
    imgs = [np.array(to_pil_image(img)) for img in images]
    imageio.mimsave('/blue/mingjieliu/jiruijin/pytorch/VAE-face/outputs/generated_images.gif', imgs)
def save_reconstructed_images(recon_images, epoch):
    save_image(recon_images.cpu(), f"/blue/mingjieliu/jiruijin/pytorch/VAE-face/outputs/output{epoch}.jpg")
def save_loss_plot(train_loss, valid_loss):
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(valid_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('/blue/mingjieliu/jiruijin/pytorch/VAE-face/outputs/loss.jpg')
    plt.show()

import torch
import torch.nn as nn
import torch.nn.functional as F
init_channels = 64 # initial number of filters
image_channels = 3 # color channels
latent_dim = 100 # number of features to consider
# define a Conv VAE
class ConvVAE(nn.Module):
    def __init__(self):
        super(ConvVAE, self).__init__()
 
        # encoder
        self.enc1 = nn.Conv2d(
            in_channels=image_channels, out_channels=init_channels, 
            kernel_size=4, stride=2, padding=2
        )
        self.enc2 = nn.Conv2d(
            in_channels=init_channels, out_channels=init_channels*2, 
            kernel_size=4, stride=2, padding=2
        )
        self.enc3 = nn.Conv2d(
            in_channels=init_channels*2, out_channels=init_channels*4, 
            kernel_size=4, stride=2, padding=2
        )
        self.enc4 = nn.Conv2d(
            in_channels=init_channels*4, out_channels=init_channels*8, 
            kernel_size=4, stride=2, padding=2
        )
        self.enc5 = nn.Conv2d(
            in_channels=init_channels*8, out_channels=1024, 
            kernel_size=4, stride=2, padding=2
        )
        self.fc1 = nn.Linear(1024, 2048)
        self.fc_mu = nn.Linear(2048, latent_dim)
        self.fc_log_var = nn.Linear(2048, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 1024)
        # decoder 
        self.dec1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=init_channels*8, 
            kernel_size=3, stride=2
        )
        self.dec2 = nn.ConvTranspose2d(
            in_channels=init_channels*8, out_channels=init_channels*4, 
            kernel_size=3, stride=2
        )
        self.dec3 = nn.ConvTranspose2d(
            in_channels=init_channels*4, out_channels=init_channels*2, 
            kernel_size=3, stride=2
        )
        self.dec4 = nn.ConvTranspose2d(
            in_channels=init_channels*2, out_channels=init_channels, 
            kernel_size=3, stride=2
        )
        self.dec5 = nn.ConvTranspose2d(
            in_channels=init_channels, out_channels=image_channels, 
            kernel_size=4, stride=2
        )
    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample
 
    def forward(self, x):
        # encoding
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = F.relu(self.enc5(x))
        batch, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        hidden = self.fc1(x)
        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        z = self.fc2(z)
        z = z.view(-1, 1024, 1, 1)
 
        # decoding
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        reconstruction = torch.sigmoid(self.dec5(x))
        return reconstruction, mu, log_var

from tqdm import tqdm
import torch 
def final_loss(bce_loss, mu, logvar):
    """
    This function will add the reconstruction loss (BCELoss) and the 
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    BCE = bce_loss 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train(model, dataloader, dataset, device, optimizer, criterion):
    model.train()
    running_loss = 0.0
    counter = 0
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
        counter += 1
        data = data
        data = data.to(device)
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(data)
        bce_loss = criterion(reconstruction, data)
        loss = final_loss(bce_loss, mu, logvar)
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
    train_loss = running_loss / counter 
    return train_loss

def validate(model, dataloader, dataset, device, criterion):
    model.eval()
    running_loss = 0.0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
            counter += 1
            data= data
            data = data.to(device)
            reconstruction, mu, logvar = model(data)
            bce_loss = criterion(reconstruction, data)
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()
        
            # save the last batch input and output of every epoch
            if i == int(len(dataset)/dataloader.batch_size) - 1:
                recon_images = reconstruction
    val_loss = running_loss / counter
    return val_loss, recon_images

import torch
import torch.optim as optim
import torch.nn as nn
# import model
import matplotlib
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
# from engine import train, validate
# from utils import save_reconstructed_images, image_to_vid, save_loss_plot
# from utils import transform
# from prepare_data import prepare_dataset
# from dataset import LFWDataset
plt.style.use('ggplot')
# define the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# a list to save all the reconstructed images in PyTorch grid format
grid_images = []

model=ConvVAE().to(device)

# define the learning parameters
lr = 0.0001
epochs = 100
batch_size = 64
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss(reduction='sum')

# initialize the transform
transform = transform()
# prepare the training and validation data loaders
train_data, valid_data = prepare_dataset(
    ROOT_PATH='/blue/mingjieliu/jiruijin/pytorch/VAE-face/input/lfw-deepfunneled/lfw-deepfunneled/'
)
trainset = LFWDataset(train_data, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
validset = LFWDataset(valid_data, transform=transform)
validloader = DataLoader(validset, batch_size=batch_size)

train_loss = []
valid_loss = []
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = train(
        model, trainloader, trainset, device, optimizer, criterion
    )
    valid_epoch_loss, recon_images = validate(
        model, validloader, validset, device, criterion
    )
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    # save the reconstructed images from the validation loop
    save_reconstructed_images(recon_images, epoch+1)
    # convert the reconstructed images to PyTorch image grid format
    image_grid = make_grid(recon_images.detach().cpu())
    grid_images.append(image_grid)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {valid_epoch_loss:.4f}")

# save the reconstructions as a .gif file
image_to_vid(grid_images)
# save the loss plots to disk
save_loss_plot(train_loss, valid_loss)
print('TRAINING COMPLETE')