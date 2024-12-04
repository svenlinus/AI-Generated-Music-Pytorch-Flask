import torch.nn as nn
import torch

class CVAE(nn.Module):
  instance = None
  def __init__(self, latent_dim=4):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim

    self.encoder = nn.Sequential(
      nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # (1, 256, 128) -> (32, 128, 64)
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # (32, 128, 64) -> (64, 64, 32)
      nn.ReLU(),
      nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (64, 64, 32) -> (128, 32, 16)
      nn.ReLU(),
      nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # (128, 32, 16) -> (256, 16, 8)
      nn.ReLU(),
    )

    self.calc_conv_output_size()
    print(f'Conv output size: {self.flatten_dim}')

    self.fc_mean = nn.Linear(self.flatten_dim, latent_dim - 2)  # -2 for concatenated emotion vector
    self.fc_log_var = nn.Linear(self.flatten_dim, latent_dim - 2)

    self.fc_decoder = nn.Linear(latent_dim, self.flatten_dim)

    self.decoder = nn.Sequential(
      nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (256, 16, 8) -> (128, 32, 16)
      nn.ReLU(),
      nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (128, 32, 16) -> (64, 64, 32)
      nn.ReLU(),
      nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (64, 64, 32) -> (32, 128, 64)
      nn.ReLU(),
      nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # (32, 128, 64) -> (1, 256, 128)
      nn.Sigmoid(),
    )

  def calc_conv_output_size(self):
    self.flatten_dim = self.encoder(torch.zeros(1, 1, 256, 128)).flatten().shape[0]

  def encode(self, x):
    h = self.encoder(x)  # Convolutional encoding
    h = torch.flatten(h, start_dim=1)  # Flatten for fully connected layers
    mean = self.fc_mean(h)
    log_var = self.fc_log_var(h)
    return mean, log_var

  def reparameterize(self, mean, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mean + eps * std

  def decode(self, z):
    h = self.fc_decoder(z)  # Fully connected layer to reshape
    h = h.view(-1, 256, 16, 8)  # Reshape for transposed convolutions
    return self.decoder(h)  # Transposed convolutional decoding

  def forward(self, x, emotion):
    mean, log_var = self.encode(x)
    z = self.reparameterize(mean, log_var)
    z_with_emotion = torch.cat((z, emotion), dim=1)
    return self.decode(z_with_emotion), mean, log_var
