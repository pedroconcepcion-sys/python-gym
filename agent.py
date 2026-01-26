import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import copy
from model_ddpg import Actor, Critic

class DDPGAgent:
    def __init__(self, state_dim, action_dim, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.005):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Redes Principales
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        
        # Redes Target (Copia de las principales para estabilidad)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        
        # Optimizadores
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.gamma = gamma
        self.tau = tau

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update_parameters(self, replay_buffer, batch_size=64):
        if replay_buffer.size < batch_size:
            return

        # 1. Muestrear el Buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # 2. Calcular el Target Q (Ecuación de Bellman)
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + (not_done * self.gamma * target_Q)

        # 3. Actualizar el Crítico (Minimizar error de predicción)
        current_Q = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q, target_Q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 4. Actualizar el Actor (Maximizar la evaluación del Crítico)
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 5. Soft Update de las redes Target
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        """Guarda los pesos del Actor y el Crítico en archivos .pth"""
        torch.save(self.actor.state_dict(), f"{filename}_actor.pth")
        torch.save(self.critic.state_dict(), f"{filename}_critic.pth")
        print(f"[AGENT] Pesos guardados como: {filename}_actor.pth y {filename}_critic.pth")

    def load(self, filename):
        """Carga pesos previamente entrenados"""
        if os.path.exists(f"{filename}_actor.pth"):
            self.actor.load_state_dict(torch.load(f"{filename}_actor.pth", map_location=self.device))
            self.critic.load_state_dict(torch.load(f"{filename}_critic.pth", map_location=self.device))
            # Sincronizamos las redes target
            self.actor_target = copy.deepcopy(self.actor)
            self.critic_target = copy.deepcopy(self.critic)
            print(f"[AGENT] Pesos cargados desde: {filename}")
        else:
            print("[AGENT] No se encontraron archivos de pesos. Iniciando desde cero.")        