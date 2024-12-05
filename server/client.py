import socket

def client_loop():
    HOST = '127.0.0.1'  # 서버의 IP 주소 (여기서는 로컬호스트)
    PORT = 30000         # 서버의 포트 번호

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        s.connect((HOST, PORT))  # 서버에 연결
        print(f"Connected to {HOST}:{PORT}")
        
        # 서버에서 받은 메시지 출력
        data = s.recv(1024)
        print(f"Received from server: {data.decode('utf-8')}")
        
    except Exception as e:
        print(f"Error connecting to {HOST}:{PORT}: {e}")
    finally:
        s.close()  # 연결 종료

if __name__ == "__main__":
    client_loop()
