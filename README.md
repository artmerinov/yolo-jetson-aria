# Realtime Object Detection using Meta Aria Glasses and Jetson AGX Orin

Realtime object detection based on [ARIA_YOLO](https://github.com/EdoWhite/ARIA_YOLO) repository. The setup includes the following:
 
- Personal laptop serving as a client
- Jetson AGX Orin serving as a server
- Meta Aria Glasses for capturing real-time video stream

Glasses are connected to laptop via USB or Wi-Fi network, and Jetson is connected to laptop with Ethernet cable or Wi-Fi network. The client sends the images to server, the server processes them through the YOLO model, and returns the model artefacts back, which are displayed on the client side. The setup follows a client-server architecture. The server waits for incoming connections, while the client can connect and start communication. This communication is handled via sockets with multithreading. The client operates with two threads: the main thread, which sends images from glasses, and an additional thread for receiving results from the server. Similarly, the server has two threads: the main one for sending results back to the client, and an additional one for receiving images.

## Usage

Server:
```bash
cd YOLO
python3 server.py \
	--yolo_weights yolo_models/yolov7-tiny.weights \
	--yolo_cfg yolo_models/yolov7-tiny.cfg
```

Client:
```bash
cd YOLO
python3 client.py \
	--interface wifi \
	--device_ip <device_ip>
```

