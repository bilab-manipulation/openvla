import cv2
import numpy as np
import time
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
import socket, pickle
from PIL import Image
import os
import json



pretrained_checkpoint = "runs/openvla-7b+arti1203_all+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
# Load Processor & Model
processor = AutoProcessor.from_pretrained(pretrained_checkpoint, trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "runs/openvla-7b+arti1203_all+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to("cuda:0")
dataset_statistics_path = os.path.join(pretrained_checkpoint, "dataset_statistics.json")
if os.path.isfile(dataset_statistics_path):
    with open(dataset_statistics_path, "r") as f:
        norm_stats = json.load(f)
    vla.norm_stats = norm_stats



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
            image = cv2.resize(image, (224,224), cv2.INTER_LANCZOS4)
            
            # 이미지가 numpy.ndarray일 때
            image = Image.fromarray(image) 

            # 변환 후 작업
            
            inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
            #NOTICE: 할 때마다 바꿔줘야함
            action = vla.predict_action(**inputs, unnorm_key="arti1203_all", do_sample=False)

            # yaw roll change
            action[5], action[4] = action[4], action[5]
            action[:-1] = action[:-1]
            print("[shlim] action", action)
            action = pickle.dumps(action)
            
            datalen = str(len(action)).zfill(10)  # 데이터 길이를 10자리로 맞춰서 전송

            # 데이터 길이를 먼저 전송
            conn.send(datalen.encode('utf-8'))
            
            print("[jslee] data sending")
            conn.send(action)

    except KeyboardInterrupt:
        print("Exiting...")
        conn.close()


if __name__ == "__main__":
    main_loop()

