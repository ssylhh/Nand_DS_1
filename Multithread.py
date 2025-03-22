# import socket
# import threading
# import time

# DEVICE_MAP = {
#     "device1": {"ip": "192.168.0.7", "ports": {4000, 5000}},
#     "device2": {"ip": "192.168.0.8", "ports": {4000, 5000}},
# }

# server_running = True  # 서버 실행 상태를 추적하는 전역 변수

# def handle_client(client_socket, addr):
#     print(f"[+] Connected: {addr}")

#     # 클라이언트로부터 장치명과 포트 번호 요청받기
#     try:
#         request_data = client_socket.recv(1024).decode().split(":")
#         if len(request_data) != 2:
#             client_socket.send(b"Invalid request format (use: device_name:port)")
#             client_socket.close()
#             return

#         device_name, port = request_data[0], int(request_data[1])

#         if device_name not in DEVICE_MAP or port not in DEVICE_MAP[device_name]["ports"]:
#             client_socket.send(b"Invalid device or port")
#             client_socket.close()
#             return

#         # 선택된 장치 및 포트 정보 가져오기
#         target_ip = DEVICE_MAP[device_name]["ip"]

#         print(f"[{addr}] Connecting to {device_name} ({target_ip}:{port})")

#         # 외부 장치와 소켓 연결
#         device_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         device_socket.connect((target_ip, port))

#         client_socket.send(b"Connected to device")

#         # 클라이언트 <-> 서버 <-> 장치 데이터 전달
#         while True:
#             try:
#                 data = client_socket.recv(1024)
#                 if not data:
#                     break
#                 device_socket.send(data)  # 장치로 데이터 전달

#                 response = device_socket.recv(1024)
#                 client_socket.send(response)  # 클라이언트로 응답 전달

#             except BlockingIOError:
#                 time.sleep(0.1)  # 데이터 처리 중 blocking 상태 방지

#     except Exception as e:
#         print(f"Error in handle_client: {e}")
#     finally:
#         client_socket.close()
#         print(f"[-] Disconnected: {addr}")

# def check_server_shutdown():
#     global server_running
#     while server_running:
#         user_input = input("Press 'q' to stop the server: ")
#         if user_input.lower() == 'q':
#             print("[*] Shutting down the server...")
#             server_running = False
#             break

# # 서버 실행
# server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# server_socket.bind(("192.168.0.11", 6000))  # 서버 IP와 포트
# server_socket.listen(5)

# print("[*] Server listening on port 6000")

# # 서버 종료 감지 스레드 실행
# shutdown_thread = threading.Thread(target=check_server_shutdown)
# shutdown_thread.daemon = True
# shutdown_thread.start()

# # 클라이언트 연결 처리
# server_socket.setblocking(False)  # accept()를 non-blocking 모드로 설정

# while server_running:
#     # accept()를 blocking 없이 실행하고, 서버가 종료되면 break
#     try:
#         client_sock, addr = server_socket.accept()
#         client_thread = threading.Thread(target=handle_client, args=(client_sock, addr))
#         client_thread.start()
#     except BlockingIOError:
#         time.sleep(0.1)  # CPU 과부하를 방지하기 위해 잠시 대기

# # 서버 종료 후 소켓 닫기
# server_socket.close()
# print("[*] Server stopped.")



# import socket
# import threading
# import time

# DEVICE_MAP = {
#     "device1": {"ip": "192.168.0.7", "ports": {4000, 5000}},
#     "device2": {"ip": "192.168.0.8", "ports": {4000, 5000}},
# }

# server_running = True  # 서버 실행 상태를 추적하는 전역 변수

# def handle_client(client_socket, addr):
#     print(f"[+] Connected: {addr}")

#     try:
#         # 클라이언트로부터 디바이스명, 포트, 그리고 디바이스의 IP와 포트 정보를 받음
#         request_data = client_socket.recv(1024).decode().split(":")
#         if len(request_data) != 4:
#             client_socket.send(b"Invalid request format (use: device_name:port:device_ip:device_port)")
#             client_socket.close()
#             return

#         device_name, port = request_data[0], int(request_data[1])
#         device_ip, device_port = request_data[2], int(request_data[3])

#         if device_name not in DEVICE_MAP or port not in DEVICE_MAP[device_name]["ports"]:
#             client_socket.send(b"Invalid device or port")
#             client_socket.close()
#             return

#         # 선택된 장치 및 포트 정보 가져오기
#         target_ip = DEVICE_MAP[device_name]["ip"]

#         print(f"[{addr}] Connecting to {device_name} ({target_ip}:{port})")
        
#         # 외부 장치와 소켓 연결
#         device_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         device_socket.connect((device_ip, device_port))  # 클라이언트에서 보낸 device_ip와 device_port로 연결
        
#         client_socket.send(b"Connected to device")

#         # 클라이언트 <-> 서버 <-> 장치 데이터 전달
#         while True:
#             try:
#                 data = client_socket.recv(1024)
#                 if not data:
#                     break
#                 device_socket.send(data)  # 장치로 데이터 전달

#                 response = device_socket.recv(1024)
#                 client_socket.send(response)  # 클라이언트로 응답 전달

#             except BlockingIOError:
#                 time.sleep(0.1)  # 데이터 처리 중 blocking 상태 방지

#     except Exception as e:
#         print(f"Error in handle_client: {e}")
#     finally:
#         client_socket.close()
#         print(f"[-] Disconnected: {addr}")

# def check_server_shutdown():
#     global server_running
#     while server_running:
#         user_input = input("Press 'q' to stop the server: ")
#         if user_input.lower() == 'q':
#             print("[*] Shutting down the server...")
#             server_running = False
#             break

# # 서버 실행
# server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# server_socket.bind(("192.168.0.11", 6000))  # 서버 IP와 포트
# server_socket.listen(5)

# print("[*] Server listening on port 6000")

# # 서버 종료 감지 스레드 실행
# shutdown_thread = threading.Thread(target=check_server_shutdown)
# shutdown_thread.daemon = True
# shutdown_thread.start()

# # 클라이언트 연결 처리
# server_socket.setblocking(False)  # accept()를 non-blocking 모드로 설정

# while server_running:
#     # accept()를 blocking 없이 실행하고, 서버가 종료되면 break
#     try:
#         client_sock, addr = server_socket.accept()
#         client_thread = threading.Thread(target=handle_client, args=(client_sock, addr))
#         client_thread.start()
#     except BlockingIOError:
#         time.sleep(0.1)  # CPU 과부하를 방지하기 위해 잠시 대기

# # 서버 종료 후 소켓 닫기
# server_socket.close()
# print("[*] Server stopped.")



# import socket
# import threading
# import time
# import select

# DEVICE_MAP = {
#     "device1": {"ip": "192.168.0.7", "ports": {4000, 5000}},
#     "device2": {"ip": "192.168.0.8", "ports": {4000, 5000}},
# }

# server_running = True  # 서버 실행 상태를 추적하는 전역 변수

# def handle_client(client_socket, addr):
#     print(f"[+] Connected: {addr}")

#     try:
#         request_data = client_socket.recv(1024).decode().split(":")
#         if len(request_data) != 4:
#             client_socket.send(b"Invalid request format (use: device_name:port:device_ip:device_port)")
#             return

#         device_name, port = request_data[0], int(request_data[1])
#         device_ip, device_port = request_data[2], int(request_data[3])

#         if device_name not in DEVICE_MAP or port not in DEVICE_MAP[device_name]["ports"]:
#             client_socket.send(b"Invalid device or port")
#             return

#         target_ip = DEVICE_MAP[device_name]["ip"]
#         print(f"[{addr}] Connecting to {device_name} ({target_ip}:{port})")

#         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as device_socket:
#             device_socket.settimeout(5)  # 5초 타임아웃 설정
#             device_socket.connect((target_ip, port))  # DEVICE_MAP의 IP와 포트로 연결
#             client_socket.send(b"Connected to device")

#             while True:
#                 data = client_socket.recv(1024)
#                 if not data:
#                     break
#                 device_socket.sendall(data)  # 장치로 데이터 전송
#                 response = device_socket.recv(1024)
#                 client_socket.sendall(response)  # 클라이언트에게 응답 전달

#     except socket.timeout:
#         client_socket.send(b"Device connection timeout")
#     except Exception as e:
#         print(f"Error in handle_client: {e}")
#     finally:
#         client_socket.close()
#         print(f"[-] Disconnected: {addr}")

# def check_server_shutdown():
#     global server_running
#     while server_running:
#         user_input = input("Press 'q' to stop the server: ")
#         if user_input.lower() == 'q':
#             print("[*] Shutting down the server...")
#             server_running = False
#             break

# server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# server_socket.bind(("192.168.0.11", 6000))
# server_socket.listen(5)

# print("[*] Server listening on port 6000")

# shutdown_thread = threading.Thread(target=check_server_shutdown, daemon=True)
# shutdown_thread.start()

# server_socket.setblocking(True)  # Blocking 모드로 변경

# while server_running:
#     readable, _, _ = select.select([server_socket], [], [], 0.5)  # CPU 부하 절감
#     if server_socket in readable:
#         client_sock, addr = server_socket.accept()
#         client_thread = threading.Thread(target=handle_client, args=(client_sock, addr))
#         client_thread.start()

# server_socket.close()
# print("[*] Server stopped.")


import socket
import threading
import time

# 디바이스 정보 매핑
DEVICE_MAP = {
    "device1": {"ip": "192.168.0.7", "ports": {4000, 5000}},
    "device2": {"ip": "192.168.0.8", "ports": {4000, 5000}},
}

# 데이터 버퍼 크기 설정 (기본 30720, 최대 30720 * 2160)
BUFFER_SIZE = 30720  

server_running = True  # 서버 실행 상태 변수
active_threads = []  # 실행 중인 스레드 목록

def forward_data(source_socket, target_socket, direction):
    """
    source_socket에서 데이터를 읽어 target_socket으로 즉시 전달
    """
    try:
        while server_running:
            data = source_socket.recv(BUFFER_SIZE)
            if not data:
                break  # 연결 종료
            target_socket.sendall(data)  # 받은 데이터를 즉시 전송
    except Exception as e:
        print(f"[!] Connection error ({direction}): {e}")
    finally:
        source_socket.close()
        target_socket.close()

def handle_client(client_socket, addr):
    """
    클라이언트와 장치 간의 실시간 데이터 중계를 설정
    """
    print(f"[+] Connected: {addr}")
    try:
        # 클라이언트로부터 요청 데이터 수신
        request_data = client_socket.recv(1024).decode().split(":")
        
        if len(request_data) != 4:
            client_socket.send(b"Invalid request format (use: device_name:port:device_ip:device_port)")
            return

        device_name, port = request_data[0], int(request_data[1])
        device_ip, device_port = request_data[2], int(request_data[3])

        if device_name not in DEVICE_MAP or port not in DEVICE_MAP[device_name]["ports"]:
            client_socket.send(b"Invalid device or port")
            return

        # 장치와 연결
        target_ip = DEVICE_MAP[device_name]["ip"]
        print(f"[{addr}] Connecting to {device_name} ({target_ip}:{port})")

        device_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        device_socket.connect((device_ip, device_port))

        client_socket.send(b"Connected to device")

        # 데이터 중계를 위한 송수신 스레드 실행
        send_thread = threading.Thread(target=forward_data, args=(client_socket, device_socket, "Client → Device"), daemon=True)
        recv_thread = threading.Thread(target=forward_data, args=(device_socket, client_socket, "Device → Client"), daemon=True)

        send_thread.start()
        recv_thread.start()

        send_thread.join()
        recv_thread.join()

    except Exception as e:
        print(f"Error handling client {addr}: {e}")
    finally:
        client_socket.close()
        print(f"[-] Disconnected: {addr}")

def check_server_shutdown():
    global server_running
    while server_running:
        user_input = input("Press 'q' to stop the server: ")
        if user_input.lower() == 'q':
            print("[*] Shutting down the server...")
            server_running = False
            break

# 서버 실행
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind(("192.168.0.11", 6000))
server_socket.listen(10)  # 동시 접속 클라이언트 증가

print("[*] Server listening on port 6000")

# 서버 종료 감지 스레드 실행
shutdown_thread = threading.Thread(target=check_server_shutdown, daemon=True)
shutdown_thread.start()

# 클라이언트 연결 처리
server_socket.setblocking(False)

while server_running:
    try:
        client_sock, addr = server_socket.accept()
        client_thread = threading.Thread(target=handle_client, args=(client_sock, addr), daemon=True)
        client_thread.start()
        active_threads.append(client_thread)
    except BlockingIOError:
        time.sleep(0.1)  # CPU 사용률 감소

# 모든 스레드 종료 대기
for thread in active_threads:
    thread.join()

server_socket.close()
print("[*] Server stopped.")
