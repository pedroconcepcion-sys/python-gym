import socket
import struct
import numpy as np
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
                        t, e, integ, last_u_from_dll, iL_raw = struct.unpack(STATE_FMT, pkt) 

                        # CALCULAR EL VOLTAJE REAL (Vout = Vref - e)
                        # Como tu Vref en PLECS es 12V fijo
                        v_out = 12.0 - e

                       # 1.1 NORMALIZACIÓN CORRECTA
                        norm_y = v_out / 12.0     # Voltaje respecto a la meta   
                        norm_e = e / 12.0         # Error respecto a la meta    
                        norm_i = integ / 1.0      # "Memoria" del error. Si llega a 1.0 es que la cosa va muy mal.

                        current_state = np.array([norm_y, norm_e, norm_i]) 

                        # 3. Acción 
                        action_raw = agent.select_action(current_state)
                        noise = np.random.normal(0, 0.05) # Ruido bajo para que no oscile por culpa nuestra
                        
                        u_intent = float(action_raw[0]) + noise
                        u = np.clip(u_intent, 0.0, 1.0) # Lo único físico: no puede ser negativo ni > 100%

                        # 4. Recompensa (Ajuste "Precisión Agresiva")
                        # Multiplicamos por 10.0. Antes un error de 4V dolía -16. Ahora dolerá -160.
                        r_voltage = -(e ** 2) * 10.0
                        
                        # PRIORIDAD 2: Estabilidad (Bajamos la exigencia).
                        # Bajamos el multiplicador de 50.0 a 5.0. 
                        # Ahora le sale "barato" mover la palanca para corregir el error.
                        if prev_action is None:
                            diff = 0.0
                        else:
                            diff = abs(u - float(prev_action[0]))
                        
                        r_stability = -(diff * 5.0) 

                        # BONUS: Castigo a la vagancia extrema
                        # Si el voltaje es bajo (<11V) y la IA tiene el motor casi apagado (u < 0.1), castigo extra.
                        r_lazy = 0.0
                        if v_out < 11.0 and u < 0.1:
                            r_lazy = -50.0

                        reward = r_voltage + r_stability + r_lazy
                        episode_reward += reward

                        # 5. Entrenar
                        if prev_state is not None:
                            buffer.add(prev_state, prev_action, current_state, reward, 0.0)
                            agent.update_parameters(buffer, batch_size=64)
                        
                        # TRACE: 
                        if step_count % 10 == 0: # Imprime cada 10 pasos 
                            print(f"[TRACE] Paso: {step_count} | T:{t:.4f}] | Vout:{v_out:.2f}V | Err:{e:.2f}V | Integ:{integ:.3f} | u:{u:.3f}")

                        # 6. Enviar a PLECS
                        conn.sendall(struct.pack(ACT_FMT, u))
                        
                        # Actualizamos punteros
                        prev_state = current_state
                        prev_action = np.array([u]) # Guardamos la acción real para la siguiente vuelta
                        step_count += 1
                        #----------------------------------------------------------------------------------
                        # 2. Recompensa
                        #reward = 5.0 if abs(e) < 0.1 else -1.0
                        #episode_reward += reward  
                        
                        # - Si Vout=5V (Error=7) -> Reward = -49.0 (¡Le duele mucho!)
                        # - Si Vout=11V (Error=1) -> Reward = -1.0 (Ya no duele tanto)
                        # r_voltage = -(e ** 2) 

                        # Bono :
                        # Si logra entrar en la zona segura (11V a 13V), le damos un premio extra
                        # para que se quiera quedar ahí a vivir.
                        #if abs(e) < 1.0:
                            #r_voltage += 10.0 
                        
                        #reward = r_voltage
                        #episode_reward += reward  # Acumulamos el puntaje
                        #----------------------------------------------------------------------------------
                        # # 3. Entrenar
                        # if prev_state is not None:
                        #     done = 0.0 
                        #     buffer.add(prev_state, prev_action, current_state, reward, done)
                        #     agent.update_parameters(buffer, batch_size=64)
                        #----------------------------------------------------------------------------------
                        # # 4. Actuar (Ruido 5%)
                        # action = agent.select_action(current_state)

                        # # Reducimos el sigma a 0.02 para ganar estabilidad física  #CAMBIO
                        # # action = (action + np.random.normal(0, 0.05)).clip(0, 1) 
                        # noise = np.random.normal(0, 0.02) #CAMBIO
                        # action = (action + noise).clip(0, 1) #CAMBIO

                        # u = float(action[0])

                        # # TRACE: 
                        # if step_count % 10 == 0: # Imprime cada 10 pasos 
                        #     print(f"[TRACE] Paso: {step_count} | T:{t:.4f}] | Vout:{v_out:.2f}V | Err:{e:.2f}V | Integ:{integ:.3f} | u:{u:.3f}")

                        # conn.sendall(struct.pack(ACT_FMT, u))
                        
                        # prev_state = current_state
                        # prev_action = action
                        # step_count += 1

                except (ConnectionError, OSError):
                    # --- DETECCIÓN DE "SUICIDIO" (DEATH PENALTY) ---
                    # Si el episodio duró menos de 950 pasos, asumimos que PLECS se cerró por error de voltaje (Kamikaze)
                    if step_count < 950 and prev_state is not None:
                        print(f"!!! MUERTE PREMATURA DETECTADA en paso {step_count} (Castigo aplicado)")
                        
                        # 1. El Castigo: Debe ser peor que sobrevivir haciéndolo mal (-1000)
                        death_penalty = -2000.0 
                        episode_reward += death_penalty

                        # 2. Grabamos el trauma en la memoria
                        # Usamos 'prev_state' como estado final, y done=1.0 (True) para marcar el final trágico
                        buffer.add(prev_state, prev_action, prev_state, death_penalty, 1.0)
                        
                        # 3. Forzamos aprendizaje inmediato
                        agent.update_parameters(buffer, batch_size=64)

                    print(f"[FIN] Simulación terminada. Recompensa Total: {episode_reward:.2f}")

                # --- GUARDADO AUTOMÁTICO (Esto sigue igual, fuera del try/except) ---
                episode_count += 1
                if episode_count % 10 == 0:
                    print(f"[INFO] Guardando checkpoint del episodio {episode_count}...")
                    agent.save("buck_controller_final")

    except KeyboardInterrupt:
        print("\n[SERVER] Deteniendo servidor y guardando estado final...")
        agent.save("buck_controller_final_detenido_manual")

if __name__ == "__main__":
    main()