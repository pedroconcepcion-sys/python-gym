import matplotlib.pyplot as plt
import numpy as np
import time

FILENAME = "training_loss.csv"

def plot_loss_live():
    print(f"Leyendo {FILENAME}...")
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    while True:
        try:
            data = np.loadtxt(FILENAME, delimiter=",")
            if data.ndim == 1: data = data.reshape(1, -1) # Fix si hay solo 1 dato
            
            # Columna 0 = Actor Loss, Columna 1 = Critic Loss
            actor_data = data[:, 0]
            critic_data = data[:, 1]
            
            # --- Gráfica 1: CRITIC LOSS (El error de predicción) ---
            ax1.clear()
            ax1.plot(critic_data, color='orange', label='Critic Loss (MSE)', alpha=0.8)
            ax1.set_title("Critic Loss (¿Qué tan sorprendido está el agente?)")
            ax1.set_ylabel("Error Cuadrático")
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # --- Gráfica 2: ACTOR LOSS (La dirección de mejora) ---
            ax2.clear()
            ax2.plot(actor_data, color='purple', label='Actor Loss', alpha=0.8)
            ax2.set_title("Actor Loss (¿Está encontrando mejores acciones?)")
            ax2.set_xlabel("Episodios")
            ax2.set_ylabel("Valor Q Negativo")
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.pause(5)
            
        except OSError:
            print("Esperando archivo...")
            time.sleep(5)
        except Exception:
            time.sleep(5)

if __name__ == "__main__":
    plot_loss_live()