import socket

def main_loop():
    HOST = '0.0.0.0'  # 로컬호스트 (자기 자신)
    PORT = 30000         # 포트 번호

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        s.bind((HOST, PORT))  # 포트에 바인딩
        s.listen(1)  # 최대 1개의 클라이언트 대기
        print(f"Server listening on {HOST}:{PORT}...")
    except Exception as e:
        print(f"Error binding to port {PORT}: {e}")
        return

    while True:
        conn, addr = s.accept()  # 클라이언트 연결 기다리기
        print(f"Connected by {addr}")
        conn.sendall(b"Hello, Client!")  # 클라이언트에게 메시지 보내기
        conn.close()  # 연결 종료

if __name__ == "__main__":
    main_loop()
