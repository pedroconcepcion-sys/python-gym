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
        if not chunk:
            raise ConnectionError("PLECS cerrado")
        data += chunk
    return data


# --- MODIFICACIÓN 1: Inicializar lista de historial ---
history_rewards = []
history_actor_loss = []
history_critic_loss = []
print(">>> Sistema de registro de aprendizaje listo.")


def main():
    episode_count = 0
    try:
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((HOST, PORT))
                s.listen(1)
                print(
                    f"\n[RL TRAINER] Esperando a PLECS en el puerto {PORT} (Episodio {episode_count + 1})...")
                conn, addr = s.accept()

                # --- REINICIO DE VARIABLES POR EPISODIO ---
                episode_reward = 0
                ep_actor_loss = 0
                ep_critic_loss = 0
                train_steps = 0
                prev_state = None
                prev_action = None
                step_count = 0

                try:
                    while True:
                        # 1. Recibir
                        pkt = recv_exact(conn, STATE_SIZE)
                        # t, e, integ, y_prev, iL = struct.unpack(STATE_FMT, pkt)
                        t, e, integ, last_u_from_dll, iL_raw = struct.unpack(
                            STATE_FMT, pkt)

                        # CALCULAR EL VOLTAJE REAL (Vout = Vref - e)
                        v_out = 12.0 - e

                        # 1.1 NORMALIZACIÓN CORRECTA
                        norm_y = v_out / 12.0     # Voltaje respecto a la meta
                        norm_e = e / 12.0         # Error respecto a la meta
                        # "Memoria" del error. Si llega a 1.0 es que la cosa va muy mal.
                        norm_i = integ / 1.0

                        current_state = np.array([norm_y, norm_e, norm_i])

                        # 2. Recompensa
                        r_voltage = -(e ** 2)
                        reward = r_voltage
                        episode_reward += reward

                        # 3. Entrenar
                        if prev_state is not None:
                            # done = 1.0 if abs(iL) > 25.0 else 0.0
                            # Desactivamos el 'done' crítico para que no corte la simulación
                            done = 0.0
                            buffer.add(prev_state, prev_action,
                                       current_state, reward, done)

                            al, cl = agent.update_parameters(
                                buffer, batch_size=64)

                            ep_actor_loss += al
                            ep_critic_loss += cl
                            train_steps += 1

                        # 4. Actuar (Ruido 5%)
                        action = agent.select_action(current_state)
                        action = (
                            action + np.random.normal(0, 0.05)).clip(0, 1)
                        u = float(action[0])

                        # TRACE:
                        if step_count % 10 == 0:  # Imprime cada 10 pasos
                            print(
                                f"[TRACE] Paso: {step_count} | T:{t:.4f}] | Vout:{v_out:.2f}V | Err:{e:.2f}V | Integ:{integ:.3f} | u:{u:.3f}")

                        conn.sendall(struct.pack(ACT_FMT, u))

                        prev_state = current_state
                        prev_action = action
                        step_count += 1

                except (ConnectionError, OSError):
                    print(
                        f"[FIN] Simulación terminada. Recompensa Total: {episode_reward:.2f}")

                    # 1. Calcular Promedios
                    if train_steps > 0:
                        avg_al = ep_actor_loss / train_steps
                        avg_cl = ep_critic_loss / train_steps
                    else:
                        avg_al, avg_cl = 0, 0

                    # 2. Guardar en Memoria
                    history_rewards.append(episode_reward)
                    history_actor_loss.append(avg_al)
                    history_critic_loss.append(avg_cl)

                    # 3. Guardar en DISCO (Dos archivos separados para orden)
                    np.savetxt("training_log.csv",
                               history_rewards, delimiter=",")

                    # Nuevo archivo con 2 columnas: Columna 0 = Actor, Columna 1 = Critic
                    np.savetxt("training_loss.csv",
                               np.column_stack(
                                   (history_actor_loss, history_critic_loss)),
                               delimiter=",")

                    print(
                        f"[FIN] Ep {episode_count + 1} | R: {episode_reward:.2f} | A_Loss: {avg_al:.4f} | C_Loss: {avg_cl:.4f}")

                # --- GUARDADO AUTOMÁTICO (Solo guardamos, no cargamos) ---
                episode_count += 1
                if episode_count % 10 == 0:
                    print(
                        f"[INFO] Guardando checkpoint del episodio {episode_count}...")
                    agent.save("buck_controller_old")

    except KeyboardInterrupt:
        print("\n[SERVER] Deteniendo servidor y guardando estado final...")
        agent.save("buck_controller_old_interrupt")


if __name__ == "__main__":
    main()
