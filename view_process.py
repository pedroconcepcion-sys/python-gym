import matplotlib.pyplot as plt
import numpy as np
import time

FILENAME = "training_log.csv"


def plot_live():
    print(f"Leyendo {FILENAME}...")
    plt.ion()  # Modo interactivo para actualizar
    fig, ax = plt.subplots(figsize=(10, 5))

    while True:
        try:
            # 1. Cargar datos
            data = np.loadtxt(FILENAME, delimiter=",")
            if data.ndim == 0:
                data = np.array([data])  # Si solo hay 1 dato

            ax.clear()

            # 2. Graficar Datos Crudos (Gris claro)
            ax.plot(data, color='lightgray',
                    label='Episodio individual', alpha=0.6)

            # 3. Graficar Tendencia (Media Móvil) - LA LÍNEA IMPORTANTE
            if len(data) >= 10:
                window = 10
                # Truco matemático para suavizar la línea
                trend = np.convolve(data, np.ones(window)/window, mode='valid')
                # Ajustamos el eje X para que coincida
                ax.plot(range(window-1, len(data)), trend, color='blue',
                        linewidth=2, label='Tendencia (Media)')

            ax.set_title(
                f"Progreso del Entrenamiento (Total Episodios: {len(data)})")
            ax.set_xlabel("Episodios")
            ax.set_ylabel("Recompensa Acumulada")
            ax.grid(True, alpha=0.3)
            ax.legend()

            plt.pause(5)  # Actualiza cada 5 segundos

        except OSError:
            print("Esperando a que se cree el archivo csv...")
            time.sleep(5)
        except Exception as e:
            print(f"Error leyendo gráfica: {e}")
            time.sleep(5)


if __name__ == "__main__":
    plot_live()
