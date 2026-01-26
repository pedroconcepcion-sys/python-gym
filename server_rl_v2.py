import socket
import struct
import numpy as np
import time
from agent import DDPGAgent
from replay_buffer import ReplayBuffer

# --- CONFIGURACIÓN GLOBAL ---
HOST, PORT = "127.0.0.1", 5555
STATE_FMT, STATE_SIZE = "<5d", struct.calcsize("<5d")
ACT_FMT, ACT_SIZE = "<1d", struct.calcsize("<1d")

# Inicialización Persistente (Cerebro nuevo en cada ejecución del script)
agent = DDPGAgent(state_dim=3, action_dim=1)
buffer = ReplayBuffer(state_dim=3, action_dim=1)

def recv_exact(conn, nbytes):
    data = b""
    while len(data) < nbytes:
        chunk = conn.recv(nbytes - len(data))
        if not chunk: raise ConnectionError("PLECS cerrado")
        data += chunk
    return data

def main():   
    episode_count = 0
    try:
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((HOST, PORT))
                s.listen(1)
                print(f"\n[RL TRAINER] Esperando a PLECS en el puerto {PORT} (Episodio {episode_count + 1})...")
                conn, addr = s.accept()

                # --- REINICIO DE VARIABLES POR EPISODIO ---
                episode_reward = 0
                prev_state = None
                prev_action = None
                step_count = 0 
                
                try:
                    while True:
                        # 1. Recibir
                        pkt = recv_exact(conn, STATE_SIZE)
                        t, e, integ, y_prev, iL = struct.unpack(STATE_FMT, pkt)
                        current_state = np.array([y_prev, e, iL])

                        # 2. Recompensa
                        r_voltage = 5.0 if abs(e) < 0.1 else -1.0
                        r_current = -0.05 * abs(iL)
                        reward = r_voltage + r_current
                        episode_reward += reward
                        
                        # 3. Entrenar
                        if prev_state is not None:
                            #done = 1.0 if abs(iL) > 25.0 else 0.0
                            # Desactivamos el 'done' crítico para que no corte la simulación
                            done = 0.0 
                            buffer.add(prev_state, prev_action, current_state, reward, done)
                            agent.update_parameters(buffer, batch_size=64)
                        
                        # 4. Actuar (Ruido 5%)
                        action = agent.select_action(current_state)
                        action = (action + np.random.normal(0, 0.05)).clip(0, 1)
                        u = float(action[0])
                        conn.sendall(struct.pack(ACT_FMT, u))
                        
                        prev_state = current_state
                        prev_action = action
                        step_count += 1

                except (ConnectionError, OSError):
                    print(f"[FIN] Simulación terminada. Recompensa Total: {episode_reward:.2f}")

                # --- GUARDADO AUTOMÁTICO (Solo guardamos, no cargamos) ---
                episode_count += 1
                if episode_count % 10 == 0:
                    print(f"[INFO] Guardando checkpoint del episodio {episode_count}...")
                    agent.save("buck_controller_12v")

    except KeyboardInterrupt:
        print("\n[SERVER] Deteniendo servidor y guardando estado final...")
        agent.save("buck_controller_12v_final")

if __name__ == "__main__":
    main()