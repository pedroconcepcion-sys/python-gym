import socket
import struct
import numpy as np
from agent import DDPGAgent

# --- CONFIGURACIÓN IDÉNTICA A PLECS ---
HOST, PORT = "127.0.0.1", 5555
STATE_FMT, STATE_SIZE = "<5d", struct.calcsize("<5d")
ACT_FMT, ACT_SIZE = "<1d", struct.calcsize("<1d")

# 1. Inicializamos al agente (Solo el Actor es necesario para inferencia)
agent = DDPGAgent(state_dim=3, action_dim=1)

# 2. CARGAMOS LOS PESOS ENTRENADOS
# Busca buck_controller_12v_actor.pth y buck_controller_12v_critic.pth
agent.load("buck_controller_12v")

def recv_exact(conn, nbytes):
    data = b""
    while len(data) < nbytes:
        chunk = conn.recv(nbytes - len(data))
        if not chunk: raise ConnectionError("PLECS cerrado")
        data += chunk
    return data

def main():
    print(f"--- MODO INFERENCIA ACTIVADO ---")
    print(f"Controlando Buck para V_out = 12V")

    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))
            s.listen(1)
            print(f"\n[CONTROLADOR] Esperando conexión de PLECS...")
            conn, addr = s.accept()
            
            try:
                while True:
                    # Recibir sensores de PLECS
                    pkt = recv_exact(conn, STATE_SIZE)
                    t, e, integ, y_prev, iL = struct.unpack(STATE_FMT, pkt)
                    
                    # Definir estado (igual que en el entrenamiento)
                    state = np.array([y_prev, e, iL])

                    # --- ACCIÓN PURA ---
                    # select_action ya nos da el valor determinista sin ruido
                    action = agent.select_action(state)
                    u = float(action[0])

                    # Enviar Duty Cycle a PLECS
                    conn.sendall(struct.pack(ACT_FMT, u))
                    
                    # Log reducido para no saturar
                    if int(t * 1e6) % 500 == 0: # Cada 500us
                        print(f"T: {t:.4f} | V_out: {y_prev:.2f}V | u: {u:.3f}")

            except (ConnectionError, OSError):
                print("[FIN] Simulación terminada.")

if __name__ == "__main__":
    main()