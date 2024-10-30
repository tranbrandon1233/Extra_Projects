import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
latent_dim = 10
hidden_size = 128
image_size = 28 * 28
learning_rate = 1e-3
batch_size = 64
epochs = 10

# Define the VAE architecture
class VAE(nn.Module):
    def __init__(self, image_size, hidden_size, latent_dim):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(image_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, latent_dim) # Mean
        self.fc22 = nn.Linear(hidden_size, latent_dim) # Log variance

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_size)
        self.fc4 = nn.Linear(hidden_size, image_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(device)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, image_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model and optimizer
model = VAE(image_size, hidden_size, latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
losses = []
for epoch in range(epochs):
  if len(losses) > 1:
      plt.title("Loss Overtime")
      plt.xlabel("Epoch")
      plt.ylabel("Loss")
      plt.plot(losses)
      plt.show()
  for i, (data, _) in enumerate(train_loader):
        data = data.to(device)

        # Forward pass
        recon_batch, mu, logvar = model(data)

        # Compute loss
        BCE = F.binary_cross_entropy(recon_batch, data.view(-1, image_size), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = BCE + KLD
        

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, epochs, i+1, len(train_loader), loss.item()))
            losses.append(loss.item())
          

# Save the trained model
torch.save(model.state_dict(), 'vae_mnist.pth')