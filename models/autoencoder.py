import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z = self.encode(x)
        o = self.decode(x)
        return z, o
    
    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)