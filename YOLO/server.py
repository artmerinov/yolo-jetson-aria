import collections
import socket
import threading
import time

import numpy as np

from aria_stream import parse_args, load_yolo, detect_objects, process_detections
from communication import send_array, recv_array


# Server constants
# SERVER_HOST = socket.gethostbyname(socket.gethostname())  # Server IP
SERVER_HOST = "192.168.1.10"  # Server IP
# SERVER_HOST = "192.168.1.36"  # Server IP
SERVER_PORT = 5050

recv_count = 0
sent_count = 0

stop_event = threading.Event()  # --> event is created in False state

# Global variables
most_recent_frame = collections.deque(maxlen=1)


def apply_model(img, net, output_layers):

    height, width, channels = img.shape
    # print(height, width, channels)

    # YOLO object detection
    outputs = detect_objects(img, net, output_layers)
    boxes, confidences, class_ids = process_detections(outputs, width, height)
    # print(boxes, confidences, class_ids)

    boxes = np.array(boxes, dtype="i8")
    confidences = np.array(confidences, dtype="f4")
    class_ids = np.array(class_ids, dtype="i8")
    # print(boxes, confidences, class_ids)

    return boxes, confidences, class_ids


def recv_thread(conn: socket.socket, stop_event: threading.Event) -> None:
    """
    Thread function to receive arrays and add them to the queue.
    """
    global most_recent_frame, recv_count

    while (1):
        tic = recv_array(conn)
        arr = recv_array(conn)

        if (tic is None) or (arr is None):
            stop_event.set()
            print(f"[{threading.current_thread().name}] Server received termination flag.", flush=True)
            break
        else:
            most_recent_frame.append((tic, arr))
            print(f"[{threading.current_thread().name}] Server received (time, image) and added to buffer.", flush=True)
            recv_count += 1


def run_server(net, output_layers):
    """
    Main server function.
    """
    global sent_count
    global recv_count
    global most_recent_frame

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind((SERVER_HOST, SERVER_PORT))
        server.listen(1)
        print(f"[{threading.current_thread().name}] Server listening on {SERVER_HOST}:{SERVER_PORT}", flush=True)

        while (1):

            conn, addr = server.accept()
            print(f"[{threading.current_thread().name}] Connected by {addr}", flush=True)

            # Recieve frames
            camera = threading.Thread(target=recv_thread, args=(conn, stop_event), name="RecvThread")
            camera.start()

            while True:
                if stop_event.is_set():
                    break

                if most_recent_frame:

                    # Added a try-except block around the data sending section
                    # to catch and handle errors related to broken connections,
                    # e.g., when a client has been disconnected, so server cannot 
                    # send message back.

                    try:
                        if len(most_recent_frame) > 0:

                            # Take the most recent frame from queue and associated time
                            tic, mrf = most_recent_frame.pop()

                            # Apply model
                            boxes, confidences, class_ids = apply_model(
                                img=mrf, 
                                net=net,
                                output_layers=output_layers
                            )

                            # Send model artefacts
                            send_array(conn, tic)
                            send_array(conn, boxes)
                            send_array(conn, confidences)
                            send_array(conn, class_ids)

                            print(f"[{threading.current_thread().name}] Server processed image and sent (tic, boxes, confidences, class_ids) to client.", flush=True)
                            sent_count += 1
                        
                        else:
                            continue


                    except (BrokenPipeError, ConnectionResetError) as e:
                        print(f"Connection error: {e}", flush=True)
                        break  # Break out of the inner loop to accept a new connection

                else:
                    time.sleep(0.1)  # Wait if queue is empty

            conn.close()
            stop_event.clear()
            print(f"[{threading.current_thread().name}] Disconnected from server. Ciao!", flush=True)

            print(f"", flush=True)
            print(f"[{threading.current_thread().name}] Statistics", flush=True)
            print(f"[{threading.current_thread().name}] sent_count={sent_count}", flush=True)
            print(f"[{threading.current_thread().name}] recv_count={recv_count}", flush=True)
            
            # Reset counters for new client connection
            sent_count = 0
            recv_count = 0
            most_recent_frame = collections.deque(maxlen=1)

            print(f"", flush=True)
            print(f"[{threading.current_thread().name}] Server listening on {SERVER_HOST}:{SERVER_PORT}", flush=True)


if __name__ == "__main__":
    np.random.seed(0)

    args = parse_args()

    # get pre-trained YOLO
    net, classes, output_layers = load_yolo(args)

    run_server(net, output_layers)
