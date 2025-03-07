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
