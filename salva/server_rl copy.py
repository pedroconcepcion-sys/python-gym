import socket
import struct
import numpy as np
from agent import DDPGAgent
from replay_buffer import ReplayBuffer

# --- CONFIGURACIÓN ---
HOST, PORT = "127.0.0.1", 5555
STATE_FMT, STATE_SIZE = "<5d", struct.calcsize("<5d")
ACT_FMT, ACT_SIZE = "<1d", struct.calcsize("<1d")

# Inicialización
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
    # OPCIONAL: Cargar progreso anterior si existe
    #agent.load("buck_controller_12v")
    prev_state = None
    prev_action = None
    episode_reward = 0
    step_count = 0

    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))
            s.listen(1)
            print(f"[RL TRAINER] Esperando a PLECS en el puerto {PORT}...")
            conn, addr = s.accept()

            episode_reward = 0
            
            try:
                while True:
                    pkt = recv_exact(conn, STATE_SIZE)
                    t, e, integ, y_prev, iL = struct.unpack(STATE_FMT, pkt)
                    
                    current_state = np.array([y_prev, e, iL])

                    # 3. FUNCIÓN DE RECOMPENSA SIN SESGO (Reward Shaping)
                    # Penalizamos el error de voltaje y el exceso de corriente
                    r_voltage = 5.0 if abs(e) < 0.1 else -1.0
                    r_current = -0.05 * abs(iL) # Penalización suave por estrés térmico/físico

                    # Cálculo de recompensa
                    reward = r_voltage + r_current
                    episode_reward += reward
                    
                    # Guardar experiencia y ENTRENAR
                    if prev_state is not None:
                        #done = 1.0 if abs(e) > 15.0 else 0.0 # Condición de fallo
                        done = 1.0 if abs(iL) > 25.0 else 0.0
                        buffer.add(prev_state, prev_action, current_state, reward, done)
                        
                        # Llamada a la optimización
                        agent.update_parameters(buffer, batch_size=64)
                    
                    # Selección de acción con exploración (añadimos un poco de ruido al principio)
                    action = agent.select_action(current_state)
                    # Opcional: añadir ruido para explorar mejor
                    #action = (action + np.random.normal(0, 0.1, size=1)).clip(0, 1)
                    action = (action + np.random.normal(0, 0.05)).clip(0, 1)
                    
                    u = float(action[0])
                    conn.sendall(struct.pack(ACT_FMT, u))
                    
                    if step_count % 100 == 0:
                        print(f"Time: {t:.4f} | Error: {e:.3f} | u: {u:.3f} | Reward: {episode_reward:.1f}")
                    
                    prev_state = current_state
                    prev_action = action
                    step_count += 1

            except (ConnectionError, OSError):
                print(f"\n[FIN] Simulación terminada. Recompensa Total: {episode_reward:.2f}")
         

if __name__ == "__main__":
    main()