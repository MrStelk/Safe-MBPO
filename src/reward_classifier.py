import torch
import torch.nn as nn
import torch.nn.functional as F

class SA(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sa_classifier = nn.Sequential(
            nn.Linear(state_dim+action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
        
    def forward(self, sa_input):
        sa_logit = self.sa_classifier(sa_input)
        return sa_logit

class SAS(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sas_classifier = nn.Sequential(
            nn.Linear(2*state_dim+action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
        
    def forward(self, sas_input):
        sas_logit = self.sas_classifier(sas_input)
        return sas_logit


class RClassifier(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.sas = SAS(state_dim, action_dim)
        self.sa = SA(state_dim, action_dim)
        self.batch_size = 256
        self.optimizer_sas = torch.optim.Adam(self.sas.parameters(), lr=lr)
        self.optimizer_sa = torch.optim.Adam(self.sa.parameters(), lr=lr)

    def step(self, sa_real, sa_virtual, sas_real, sas_virtual):
        """
        sa_real: Tensor of real (state, action) pairs, label=1
        sa_virtual: Tensor of virtual (state, action) pairs, label=0
        sas_real: Tensor of real (state, action, next_state) triples, label=1
        sas_virtual: Tensor of virtual (state, action, next_state) triples, label=0
        """
        # Prepare inputs and labels
        sa_inputs = torch.cat([sa_real, sa_virtual], dim=0)
        sa_labels = torch.cat([torch.ones(len(sa_real)), torch.zeros(len(sa_virtual))], dim=0).long()

        sas_inputs = torch.cat([sas_real, sas_virtual], dim=0)
        sas_labels = torch.cat([torch.ones(len(sas_real)), torch.zeros(len(sas_virtual))], dim=0).long()

        # Forward pass
        sa_logits = self.sa(sa_inputs)
        sas_logits = self.sas(sas_inputs)

        # Compute loss
        loss_sa = self.criterion(sa_logits, sa_labels)
        loss_sas = self.criterion(sas_logits, sas_labels)

        # Backpropagation
        self.optimizer_sa.zero_grad()
        loss_sa.backward()
        self.optimizer_sa.step()

        self.optimizer_sas.zero_grad()
        loss_sas.backward()
        self.optimizer_sas.step()

        return {"loss_sa": loss_sa.item(), "loss_sas": loss_sas.item()}
        
