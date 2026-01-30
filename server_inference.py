import socket
import struct
import numpy as np
from agent import DDPGAgent

# --- CONFIGURACIÓN ---
HOST, PORT = "127.0.0.1", 5555
STATE_FMT, STATE_SIZE = "<5d", struct.calcsize("<5d")
ACT_FMT, ACT_SIZE = "<1d", struct.calcsize("<1d")

# 1. Cargar el Agente
agent = DDPGAgent(state_dim=3, action_dim=1)
try:
    # Cargar los pesos entrenados
    agent.load("buck_controller_result_finetune")
    print("¡Cerebro cargado y listo!")
except:
    print("Error: No encuentro los pesos .pth")
    exit()


def recv_exact(conn, nbytes):
    data = b""
    while len(data) < nbytes:
        chunk = conn.recv(nbytes - len(data))
        if not chunk:
            raise ConnectionError("PLECS cerrado")
        data += chunk
    return data


def main():
    print(f"--- MODO INFERENCIA: ARRANQUE SUAVE + INTEGRADOR LIMPIO ---")

    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))
            s.listen(1)
            print(f"\n[CONTROLADOR] Esperando a PLECS...")
            conn, addr = s.accept()

            step_count = 0

            # VARIABLES DE CONTROL SUAVE
            prev_u = 0.0        # Para el filtro de suavizado
            # Nuestro integrador limpio en Python (La Solución)
            py_integ = 0.0
            last_time = 0.0     # Para calcular el tiempo delta (dt)

            try:
                while True:
                    pkt = recv_exact(conn, STATE_SIZE)
                    # Recibimos datos, pero IGNORAMOS el 'integ' que viene de PLECS
                    t, e_plecs, integ_plecs_sucio, last_u_plecs, iL_raw = struct.unpack(
                        STATE_FMT, pkt)

                    # 1. CÁLCULO DE DT (Tiempo entre pasos para integrar bien)
                    dt = t - last_time
                    if dt < 0:
                        dt = 0
                    last_time = t

                    # 2. CÁLCULO DEL VOLTAJE
                    v_out = 12.0 - e_plecs

                    # 3. GESTIÓN DEL INTEGRADOR (EL TRUCO DE LA AMNESIA)
                    # Solo empezamos a sumar error cuando el Soft Start termina (Paso 200)
                    if step_count >= 200:
                        py_integ += e_plecs * dt  # Integramos nosotros mismos
                    else:
                        py_integ = 0.0  # Mente en blanco durante el arranque

                    # 4. PREPARAR ESTADO (Usando nuestro Integrador Limpio)
                    norm_v = v_out / 12.0
                    norm_e = e_plecs / 12.0
                    norm_i = py_integ / 1.0  # Usamos nuestra variable limpia

                    state = np.array([norm_v, norm_e, norm_i])

                    # 5. DECISIÓN DE LA IA
                    action = agent.select_action(state)
                    ai_u = float(action[0])

                    target_u = ai_u  # IA 100%

                    u = 0.98 * prev_u + 0.02 * target_u
                    prev_u = u

                    # Confianza total en la red neuronal
                    # u = target_u

                    # Enviar
                    conn.sendall(struct.pack(ACT_FMT, u))

                    if step_count % 100 == 0:
                        print(
                            f"T:{t:.4f} | Vout:{v_out:.2f}V | IA_u:{ai_u:.2f} | Final_u:{u:.3f}")

                    step_count += 1

            except (ConnectionError, OSError):
                print("[FIN]")


if __name__ == "__main__":
    main()
