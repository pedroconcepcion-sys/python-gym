import torch
import torch.nn as nn
import torch.nn.functional as F


# NOTA MIA: Este archivo contiene la "inteligencia".
# Definimos el Actor (el que decide el PWM) y el Crítico (el que juzga si la decisión fue buena).

class Actor(nn.Module):
    def __init__(self, state_dim=2, action_dim=1):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.output = nn.Linear(64, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Sigmoid garantiza que el Duty Cycle 'u' esté entre 0.0 y 1.0
        return torch.sigmoid(self.output(x))

class Critic(nn.Module):
    def __init__(self, state_dim=2, action_dim=1):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.output = nn.Linear(64, 1)
        
    def forward(self, state, action):
        # El crítico evalúa el par (Estado, Acción)
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output(x)