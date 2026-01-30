import xmlrpc.client
import time

# Configuración del servidor RPC de PLECS
PLECS_RPC_URL = "http://localhost:1080/RPC2"
server = xmlrpc.client.ServerProxy(PLECS_RPC_URL)

MODEL_NAME = "dll_block_2_v1"


def run_training(num_episodes):
    print(f"--- Iniciando entrenamiento: {num_episodes} episodios ---")

    for episode in range(num_episodes):
        try:
            print(f"\n[CONTROL] Lanzando Episodio {episode + 1}...")

            # Usamos el prefijo 'plecs.' porque tu consola confirmó que así se llama el método
            server.plecs.simulate(MODEL_NAME)

            print(f"[CONTROL] Episodio {episode + 1} finalizado con éxito.")
        except Exception:
            # IGNORAMOS el error de PLECS y seguimos.
            # Esto evita que el script se detenga si hay una discontinuidad.
            pass
        time.sleep(0.5)


if __name__ == "__main__":
    run_training(500)  # Ejecuta X episodios de entrenamiento
