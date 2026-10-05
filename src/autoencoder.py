import copy
import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """
    Deep Autoencoder for Network Flow Anomaly Detection.
    Follows Fed-ANIDS specification:
    Encoder: Input Dim -> 64 -> 32 (Latent space)
    Decoder: 32 -> 64 -> Input Dim
    Uses LayerNorm to remain mathematically stable across small client batches
    in federated learning settings.
    """
    def __init__(self, input_dim, latent_dim=32, hidden_dim=64):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Encoder Network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU()
        )

        # Decoder Network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon

    def compute_reconstruction_errors(self, x, batch_size=256, device="cpu"):
        """
        Computes per-sample Mean Squared Error (MSE) reconstruction loss:
        e_i = (1 / D) * sum_{j=1}^D (x_{ij} - x'_ij)^2
        """
        self.eval()
        self.to(device)
        errors = []

        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                batch = torch.tensor(x[i : i + batch_size], dtype=torch.float32, device=device)
                batch_recon = self.forward(batch)
                sample_errors = torch.mean((batch - batch_recon) ** 2, dim=1)
                errors.extend(sample_errors.cpu().numpy())

        return errors

    def get_weights(self):
        """Returns a deepcopy of model state dictionary."""
        return copy.deepcopy(self.state_dict())

    def set_weights(self, state_dict):
        """Loads state dictionary into model."""
        self.load_state_dict(copy.deepcopy(state_dict))
