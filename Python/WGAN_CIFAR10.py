import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
import os
from google.colab import drive
drive.mount('/content/drive')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Hyperparameters
latent_dim = 100
image_size = 64
batch_size = 64
n_critic = 5
clip_value = 0.01
lr = 0.00005
n_epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reusable blocks for repeated layers
class ResBlock(nn.Module):
    def __init__(self, channels, num_layers=5):
        super(ResBlock, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(True)
            ])
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return x + self.layers(x)  # Residual connection

class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_generator=False):
        super(TransitionBlock, self).__init__()
        if is_generator:
            self.transition = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            )
        else:
            self.transition = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
            
    def forward(self, x):
        return self.transition(x)

# Generator Network
class Generator(nn.Module):
    def __init__(self, layers_per_block=5):
        super(Generator, self).__init__()
        
        # Initial block
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        
        # Deep residual blocks with transitions
        self.blocks = nn.ModuleList([
            # 512 channels block
            ResBlock(512, layers_per_block),
            TransitionBlock(512, 256, is_generator=True),
            
            # 256 channels block
            ResBlock(256, layers_per_block),
            TransitionBlock(256, 128, is_generator=True),
            
            # 128 channels block
            ResBlock(128, layers_per_block),
            TransitionBlock(128, 64, is_generator=True),
            
            # 64 channels block
            ResBlock(64, layers_per_block),
        ])
        
        # Final output layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)
        for block in self.blocks:
            x = block(x)
        return self.final(x)

# Critic Network
class Critic(nn.Module):
    def __init__(self, layers_per_block=5):
        super(Critic, self).__init__()
        
        # Initial block
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Deep residual blocks with transitions
        self.blocks = nn.ModuleList([
            # 64 channels block
            ResBlock(64, layers_per_block),
            TransitionBlock(64, 128, is_generator=False),
            
            # 128 channels block
            ResBlock(128, layers_per_block),
            TransitionBlock(128, 256, is_generator=False),
            
            # 256 channels block
            ResBlock(256, layers_per_block),
            TransitionBlock(256, 512, is_generator=False),
            
            # 512 channels block
            ResBlock(512, layers_per_block),
        ])
        
        # Final output layer
        self.final = nn.Conv2d(512, 1, 4, 1, 0, bias=False)

    def forward(self, x):
        x = self.initial(x)
        for block in self.blocks:
            x = block(x)
        return self.final(x).view(-1)


# Load CIFAR10 dataset
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalized for RGB
])

dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# Initialize networks and optimizers
generator = Generator(num_layers=10).to(device)
critic = Critic(num_layers=10).to(device)

if 'g_model.pth' in os.listdir('/content/drive/MyDrive/WGAN'):
    generator.load_state_dict(torch.load('/content/drive/MyDrive/WGAN/g_model.pth', weights_only=True))
    generator.eval()

if 'c_model.pth' in os.listdir('/content/drive/MyDrive/WGAN'):
    critic.load_state_dict(torch.load('/content/drive/MyDrive/WGAN/c_model.pth', weights_only=True))
    critic.eval()
g_optimizer = optim.RMSprop(generator.parameters(), lr=lr)
c_optimizer = optim.RMSprop(critic.parameters(), lr=lr)

# Function to generate and display images
def display_images(generator, fixed_noise, epoch, batch):
    with torch.no_grad():
        fake_images = generator(fixed_noise).cpu()
        plt.figure(figsize=(10, 10))
        plt.title(f"Generated Images (Epoch {epoch}, Batch {batch})")
        plt.imshow(np.transpose(vutils.make_grid(fake_images, padding=2, normalize=True), (1, 2, 0)))
        plt.axis('off')
        plt.show()
        plt.close()

# Training loop
fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)
generator.train()
critic.train()

# saved_gradients = []

for epoch in range(n_epochs):
    for batch_idx, (real_imgs, _) in enumerate(dataloader):
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)

        # Train Critic
        for _ in range(n_critic):
          c_optimizer.zero_grad()

          # Generate fake images
          z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
          fake_imgs = generator(z)

          # Calculate critic loss
          critic_real = critic(real_imgs).mean()
          critic_fake = critic(fake_imgs.detach()).mean()
          c_loss = critic_fake - critic_real

          c_loss.backward()
          # Store gradients
          # for param in critic.parameters():
          #   if param.grad is not None:
          #      saved_gradients.append(param.grad.clone())
          # print(saved_gradients)
          c_optimizer.step()

          # Clip critic weights
          for p in critic.parameters():
              p.data.clamp_(-clip_value, clip_value)

        # Train Generator
        g_optimizer.zero_grad()

        # Generate fake images
        z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake_imgs = generator(z)

        # Calculate generator loss
        g_loss = -critic(fake_imgs).mean()

        g_loss.backward()
        g_optimizer.step()

        # Display progress
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/{n_epochs}] Batch [{batch_idx}/{len(dataloader)}] "
                  f"C_loss: {c_loss.item():.4f} G_loss: {g_loss.item():.4f}")
            display_images(generator, fixed_noise, epoch, batch_idx)
            torch.save(generator.state_dict(), '/content/drive/MyDrive/WGAN/g_model.pth')
            torch.save(critic.state_dict(), '/content/drive/MyDrive/WGAN/c_model.pth')


print("Training finished!")

# Generate final sample of images
display_images(generator, fixed_noise, n_epochs, "Final")