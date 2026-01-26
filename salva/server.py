import socket
import struct

HOST = "127.0.0.1"
PORT = 5555

# PLECS DLL -> Python: 4 doubles (t, e, i, y_prev)  -  envía bytes crudos el DLL en C 
# Recibes 32 bytes en total ( 4*8 bytes).

STATE_FMT = "<4d"
STATE_SIZE = struct.calcsize(STATE_FMT)

# Python -> PLECS DLL: 1 double (u)
ACT_FMT = "<1d"
ACT_SIZE = struct.calcsize(ACT_FMT)

# Necesitamos asegurarnos de recibir exactamente nbytes bytes
def recv_exact(conn: socket.socket, nbytes: int) -> bytes:
    data = b""
    while len(data) < nbytes:
        chunk = conn.recv(nbytes - len(data))
        if not chunk:
            raise ConnectionError("Socket closed by peer")
        data += chunk
    return data

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"[PY] Listening on {HOST}:{PORT} ...")

        conn, addr = s.accept()
        print(f"[PY] Connected from {addr}")

    try:
        while True:
            pkt = recv_exact(conn, STATE_SIZE)
            t, e, integ, y_prev = struct.unpack(STATE_FMT, pkt)
            print(f"[PY] t={t:.6f}  e={e:.6f}  i={integ:.6f}  y_prev={y_prev:.6f}")
            
            # Ganancia proporcional
            Kp = 0.1

            u = Kp * e
            u = max(0.0, min(1.0, u))
            print(f"[PY] Control action u={u:.6f}")
            
            conn.sendall(struct.pack(ACT_FMT, u))

    except (ConnectionError, ConnectionResetError, OSError) as ex:
        print(f"[PY] Disconnected: {ex}")
        print("[PY] Simulation ended, exiting.")


if __name__ == "__main__":
    main()
