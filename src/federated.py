import copy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from src.data_loader import get_dataloader


class FederatedClient:
    """
    Simulates a localized client node (e.g., enterprise organization).
    Trains the Autoencoder strictly on its local private benign network traffic.
    Supports both standard local updates (FedAvg) and proximal updates (FedProx).
    """
    def __init__(self, client_id, local_data, model_template, device="cpu"):
        self.client_id = client_id
        self.local_data = local_data
        self.num_samples = len(local_data)
        self.model = copy.deepcopy(model_template)
        self.device = device

    def train(self, global_weights, epochs=3, batch_size=64, lr=0.001, strategy="fedavg", mu=0.01):
        """
        Trains local model on private benign data.
        strategy: 'fedavg' or 'fedprox'
        mu: proximal penalty coefficient for FedProx
        """
        self.model.set_weights(global_weights)
        self.model.to(self.device)
        self.model.train()

        optimizer = Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        dataloader = get_dataloader(self.local_data, batch_size=batch_size, shuffle=True)

        epoch_losses = []

        # Keep a copy of initial global parameters for FedProx proximal regularizer
        if strategy.lower() == "fedprox" and mu > 0:
            global_params = [p.detach().clone() for p in self.model.parameters()]

        for epoch in range(epochs):
            batch_losses = []
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()

                recon_x = self.model(batch_x)
                loss = criterion(recon_x, batch_x)

                # FedProx proximal regularization: (mu / 2) * ||w - w_t||^2
                if strategy.lower() == "fedprox" and mu > 0:
                    proximal_term = 0.0
                    for param, g_param in zip(self.model.parameters(), global_params):
                        proximal_term += torch.sum((param - g_param) ** 2)
                    loss += (mu / 2.0) * proximal_term

                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())

            epoch_losses.append(np.mean(batch_losses))

        avg_loss = float(np.mean(epoch_losses))
        return self.model.get_weights(), self.num_samples, avg_loss


class FederatedServer:
    """
    Central Orchestrator:
    - Broadcasts global Autoencoder weights.
    - Aggregates client updates using weighted Federated Averaging.
    - Tracks convergence across rounds.
    """
    def __init__(self, global_model, val_data, device="cpu"):
        self.global_model = global_model
        self.val_data = val_data
        self.device = device
        self.history = {
            "round": [],
            "train_loss": [],
            "val_loss": []
        }

    def aggregate(self, client_updates):
        """
        Performs weighted parameter averaging:
        w_{t+1} = sum_k (n_k / N) * w_k
        """
        total_samples = sum(num_samples for _, num_samples, _ in client_updates)
        aggregated_weights = copy.deepcopy(client_updates[0][0])

        # Zero out aggregated state dict
        for key in aggregated_weights.keys():
            aggregated_weights[key] = torch.zeros_like(aggregated_weights[key], dtype=torch.float32)

        # Accumulate weighted sums
        for weights, num_samples, _ in client_updates:
            weight_factor = num_samples / total_samples
            for key in aggregated_weights.keys():
                aggregated_weights[key] += weights[key].float() * weight_factor

        self.global_model.set_weights(aggregated_weights)
        return aggregated_weights

    def evaluate_val_loss(self):
        """Calculates reconstruction loss on the benign validation dataset."""
        self.global_model.eval()
        self.global_model.to(self.device)
        criterion = nn.MSELoss()
        dataloader = get_dataloader(self.val_data, batch_size=256, shuffle=False)
        total_loss = 0.0
        with torch.no_grad():
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(self.device)
                recon = self.global_model(batch_x)
                total_loss += criterion(recon, batch_x).item() * len(batch_x)
        return total_loss / len(self.val_data)

    def train_federated(self, clients, rounds=15, client_fraction=1.0, local_epochs=3,
                        lr=0.001, strategy="fedavg", mu=0.01):
        """
        Executes full federated learning communication rounds.
        """
        print(f"\n=======================================================")
        print(f"[*] Starting Federated Training: {strategy.upper()}")
        print(f"    - Number of Clients: {len(clients)}")
        print(f"    - Communication Rounds: {rounds}")
        print(f"    - Local Epochs per Round: {local_epochs}")
        print(f"    - Learning Rate: {lr}")
        if strategy.lower() == "fedprox":
            print(f"    - FedProx Mu (Penalty): {mu}")
        print(f"=======================================================")

        num_selected = max(1, int(len(clients) * client_fraction))

        for r in range(1, rounds + 1):
            # Select clients for this round
            selected_indices = np.random.choice(len(clients), size=num_selected, replace=False)
            current_weights = self.global_model.get_weights()

            client_updates = []
            round_train_losses = []

            for idx in selected_indices:
                client = clients[idx]
                weights, n_samples, loss = client.train(
                    global_weights=current_weights,
                    epochs=local_epochs,
                    lr=lr,
                    strategy=strategy,
                    mu=mu
                )
                client_updates.append((weights, n_samples, loss))
                round_train_losses.append(loss)

            # Server aggregates updates
            self.aggregate(client_updates)

            # Evaluate global model on benign validation data
            avg_train_loss = float(np.mean(round_train_losses))
            val_loss = self.evaluate_val_loss()

            self.history["round"].append(r)
            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(val_loss)

            print(f" Round {r:02d}/{rounds:02d} | Avg Train MSE: {avg_train_loss:.6f} | Val MSE: {val_loss:.6f}")

        print(f"[+] Federated training completed successfully for {strategy.upper()}.\n")
        return self.history
