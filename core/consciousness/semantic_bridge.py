import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger("Consciousness.SemanticBridge")

# 🔒 [M5 64GB] CPU threads for semantic bridge operations
torch.set_num_threads(6)
DEVICE = torch.device("cpu")

class LatentProjector(nn.Module):
    """
    Projector that maps high-dim embeddings into the latent manifold.
    Evolution 23.1: Stabilized for Apple Silicon unified memory.
    """
    def __init__(self, input_dim=764, latent_dim=128):
        super().__init__()
        self.projector = nn.Linear(input_dim, latent_dim, bias=False)
        self.device = DEVICE
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 🔒 [Grok Edit] Phase 1: 2% Gaussian noise to stabilize gradients
        if self.training:
            noise = torch.randn_like(x) * 0.02
            x = x + noise
        
        return torch.tanh(self.projector(x))

class BridgeTrainer:
    """
    Handles calibration of the LatentProjector.
    """
    def __init__(self, model: LatentProjector, lr=1e-3):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
    def step(self, x: torch.Tensor, target: torch.Tensor):
        self.optimizer.zero_grad()
        output = self.model(x)
        
        # Standard MSE Loss
        mse_loss = F.mse_loss(output, target)
        
        # 🔗 [Grok Edit] Evolution 2: Orthogonal Loss
        # Prevents virtual token collapse by forcing weight vectors to be orthogonal
        W = self.model.projector.weight
        ortho_loss = torch.norm(torch.mm(W, W.t()) - torch.eye(W.size(0), device=DEVICE))
        
        total_loss = mse_loss + 0.1 * ortho_loss
        
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
