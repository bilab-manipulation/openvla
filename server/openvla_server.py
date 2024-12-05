import cv2
import numpy as np
import time
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
import socket, pickle


# Load Processor & Model
processor = AutoProcessor.from_pretrained("runs/openvla-7b+arti1203_all+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "runs/openvla-7b+arti1203_all+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to("cuda:0")


def recvall(sock):
    datalen = int(sock.recv(10).decode("utf-8"))
    print("[jslee] received data length: ", datalen)
    fragments = []
    while datalen:
        chunk = sock.recv(4096)
        fragments.append(chunk)
        datalen -= len(chunk)
    return b''.join(fragments)


def main_loop():
    HOST = '0.0.0.0'
    PORT = 30000
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen(1)
    conn, addr = s.accept()
    print ('Connected by', addr)

    try: 
        while True:
            #get data from client
            print("[jslee] data receiving")
            data = recvall(conn)
            image, prompt = pickle.loads(data)
            print("[jslee] data received :",  prompt)

            inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
            #NOTICE: 할 때마다 바꿔줘야함
            action = vla.predict_action(**inputs, unnorm_key="arti1203_all", do_sample=False)

            # yaw roll change
            action[5], action[4] = action[4], action[5]
            action = pickle.dumps(action)
            
            print("[jslee] data sending")
            conn.send(action)

    except KeyboardInterrupt:
        print("Exiting...")
        conn.close()


if __name__ == "__main__":
    main_loop()

