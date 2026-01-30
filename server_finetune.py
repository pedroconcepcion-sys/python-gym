import socket
import struct
import numpy as np
from agent import DDPGAgent
from replay_buffer import ReplayBuffer

# --- CONFIGURACIÓN GLOBAL ---
HOST, PORT = "127.0.0.1", 5555
STATE_FMT, STATE_SIZE = "<5d", struct.calcsize("<5d")
ACT_FMT, ACT_SIZE = "<1d", struct.calcsize("<1d")

agent = DDPGAgent(state_dim=3, action_dim=1)

# 1. CARGAMOS AL CAMPEÓN VIGENTE
# Asegúrate de que este nombre coincida con el archivo que generó el 11.95V
try:
    nombre_pesos = "buck_controller_old"
    agent.load(nombre_pesos) 
    print(">>> ¡Cerebro cargado! Iniciando fase de estabilización...")
except:
    print(">>> ERROR: No encuentro los pesos " + nombre_pesos)
    exit()

buffer = ReplayBuffer(state_dim=3, action_dim=1)

def recv_exact(conn, nbytes):
    data = b""
    while len(data) < nbytes:
        chunk = conn.recv(nbytes - len(data))
        if not chunk: raise ConnectionError("PLECS cerrado")
        data += chunk
    return data

def main():   
    episode_offset = 200 # Solo para llevar la cuenta visual
    episode_count = 0
    
    try:
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((HOST, PORT))
                s.listen(1)
                
                current_ep = episode_offset + episode_count + 1
                print(f"\n[FINETUNE STABLE] Esperando a PLECS (Episodio {current_ep})...")
                conn, addr = s.accept()

                episode_reward = 0
                prev_state = None
                prev_action = None
                step_count = 0 
                
                try:
                    while True:
                        # 1. RECIBIR
                        pkt = recv_exact(conn, STATE_SIZE)
                        t, e, integ, last_u_from_dll, iL_raw = struct.unpack(STATE_FMT, pkt)

                        # 2. ESTADO (MANTENER IDÉNTICO AL QUE FUNCIONÓ)
                        norm_u = last_u_from_dll       
                        norm_e = e / 12.0              
                        norm_i = iL_raw / 10.0         
                        current_state = np.array([norm_u, norm_e, norm_i])

                        # 3. ACCIÓN (Primero decidimos, luego castigamos)
                        action = agent.select_action(current_state)
                        
                        # Ruido muy bajo (0.02) solo para que pruebe suavizar
                        noise = np.random.normal(0, 0.02) 
                        action = (action + noise).clip(0, 1)
                        u = float(action[0])

                        # 4. RECOMPENSA CON TU LÓGICA NUEVA
                        
                        # A) Voltaje: Mantenemos lo que funciona
                        r_voltage = -(e ** 2)

                        # B) Estabilidad: Tu código exacto
                        if prev_action is None:
                            diff_u = 0.0
                        else:
                            # Calculamos diferencia real
                            diff_u = abs(u - float(prev_action[0]))

                        # Castigo moderado (5.0) para pulir sin congelar
                        r_stability = -(diff_u * 5.0) 

                        reward = r_voltage + r_stability
                        episode_reward += reward

                        # 5. ENTRENAR
                        if prev_state is not None:
                            buffer.add(prev_state, prev_action, current_state, reward, 0.0)
                            agent.update_parameters(buffer, batch_size=64)
                           
                        
                        # 6. ENVIAR
                        conn.sendall(struct.pack(ACT_FMT, u))
                        
                        # Trace
                        if step_count % 100 == 0: 
                            v_out = 12.0 - e
                            print(f"[Paso {step_count}] Vout:{v_out:.2f}V | u:{u:.3f} | StableCost: {r_stability:.2f}")

                        prev_state = current_state
                        prev_action = action 
                        step_count += 1

                except (ConnectionError, OSError):
                    print(f"[FIN] Ep {current_ep} | Recompensa: {episode_reward:.2f}")

                # GUARDADO CON NOMBRE NUEVO
                episode_count += 1
                if episode_count % 10 == 0:
                    print(f"[INFO] Guardando 'buck_controller_result_finetune'...")
                    agent.save("buck_controller_result_finetune")

    except KeyboardInterrupt:
        print("\n[SERVER] Guardando y cerrando...")
        agent.save("buck_controller_result_finetune_imcomplete")

if __name__ == "__main__":
    main()