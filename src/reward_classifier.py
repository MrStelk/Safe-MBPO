import torch
import torch.nn as nn
import torch.nn.functional as F
from .torch_util import Module, device
from .config import BaseConfig, Configurable

class SA(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sa_classifier = nn.Sequential(
            nn.Linear(state_dim+action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, sa_input):
        sa_logit = self.sa_classifier(sa_input)
        return sa_logit

class SAS(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sas_classifier = nn.Sequential(
            nn.Linear(2*state_dim+action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, sas_input):
        sas_logit = self.sas_classifier(sas_input)
        return sas_logit


class RClassifier(Module, Configurable):
    class Config(BaseConfig):
        sas_hidden_dim = 200
        sa_hidden_dim = 200
        batch_size = 256
        learning_rate = 1e-3
    
    def __init__(self, config, state_dim, action_dim, hidden_dim=64):
        Configurable.__init__(self, config)
        Module.__init__(self)
        
        self.sas = SAS(state_dim, action_dim, self.sas_hidden_dim)
        self.sa = SA(state_dim, action_dim, self.sa_hidden_dim)
        self.optimizer_sas = torch.optim.Adam(self.sas.parameters(), lr=self.learning_rate)
        self.optimizer_sa = torch.optim.Adam(self.sa.parameters(), lr=self.learning_rate)

        self.criterion = nn.BCELoss()

         self.register_buffer('total_updates', torch.zeros([]))

    def step(self, sa_real, sa_virtual, sas_real, sas_virtual):
        """
        sa_real: Tensor of real (state, action) pairs, label=1
        sa_virtual: Tensor of virtual (state, action) pairs, label=0
        sas_real: Tensor of real (state, action, next_state) triples, label=1
        sas_virtual: Tensor of virtual (state, action, next_state) triples, label=0
        """
        # Prepare inputs and labels
        sa_inputs = torch.cat([sa_real, sa_virtual], dim=0)
        sa_labels = torch.cat([torch.ones(sa_real.shape[0]), torch.zeros(sa_virtual.shape[0])], dim=0).reshape(-1,1).to(device)

        sas_inputs = torch.cat([sas_real, sas_virtual], dim=0)
        sas_labels = torch.cat([torch.ones(sas_real.shape[0]), torch.zeros(sas_virtual.shape[0])], dim=0).reshape(-1,1).to(device)
        
        #print(sa_inputs.shape, sa_labels.shape)
        #print(sas_inputs.shape, sas_labels.shape)

        # Forward pass
        sa_logits = self.sa(sa_inputs)
        sas_logits = self.sas(sas_inputs)

        #print(sa_logits.shape, sas_logits.shape)

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

        self.total_updates += 1

        return {"loss_sa": loss_sa.item(), "loss_sas": loss_sas.item()}
        
