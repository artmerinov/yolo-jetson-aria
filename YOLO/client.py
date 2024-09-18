import argparse
import collections
import socket
import numpy as np
import threading
import time
import cv2
import aria.sdk as aria

from communication import recv_array, send_array
from aria_stream import device_stream, device_subscribe, draw_labels_and_boxes, quit_keypress


# Client constants
SERVER_HOST = "192.168.1.10"  # Server IP
# SERVER_HOST = "192.168.1.23"  # Server IP
# SERVER_HOST = "192.168.1.36"  # Server IP
SERVER_PORT = 5050

# Global counters for debug
recv_count = 0
sent_count = 0
stop_event = threading.Event()  # --> event is created in False state

# most_recent_buff is a queue that stores tuples 
# of the most recent images from the Aria glasses 
# with its timestamps sent by client to server 
most_recent_buff = collections.deque(maxlen=128)

# most_recent_bbox is a queue that stores a tuple
# of the most recent result recieved from the server.
# This results is a tuple of timestamp, bounding_boxes, 
# confidences, and class_ids.
most_recent_bbox = collections.deque(maxlen=1)


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--interface",
        dest="streaming_interface",
        type=str,
        required=True,
        help="Type of interface to use for streaming. Options are usb or wifi.",
        choices=["usb", "wifi"],
    )
    parser.add_argument(
        "--profile",
        dest="profile_name",
        type=str,
        default="profile18",
        required=False,
        help="Profile to be used for streaming.",
    )
    parser.add_argument(
        "--device_ip", 
        help="IP address to connect to the device over wifi"
    )

    return parser.parse_args()


def recv_thread(conn: socket.socket, 
                stop_event: threading.Event,
                ) -> None:
    """
    Thread function to recieve result from the server.
    """
    global recv_count
    global most_recent_bbox


    while (1):

        if stop_event.is_set():
            break
        
        recv_ts = recv_array(conn)
        recb_boxes = recv_array(conn)
        recv_confidences = recv_array(conn)
        recv_class_ids = recv_array(conn)

        if ((recv_ts is None) or (recb_boxes is None) or (recv_confidences is None) or (recv_class_ids is None)):
            print(f"[{threading.current_thread().name}] Client received None from server. Skipping.", flush=True)
            continue
        
        print(f"[{threading.current_thread().name}] Client received (ts, boxes, confidences, class_ids) from server.", flush=True)
        recv_count += 1

        # Store recieved data
        most_recent_bbox.append((recv_ts, recb_boxes, recv_confidences, recv_class_ids))


def run_client(classes):
    """
    Main client function.
    """
    global recv_count
    global sent_count
    global most_recent_buff
    global most_recent_bbox

    # Get camera info
    print("CONNECT CAMERA...", flush=True)
    streaming_manager, streaming_client, device_client, device = device_stream(args)
    observer = device_subscribe(streaming_client)
    print("CAMERA IS READY FOR STREAMING THROUGH SOCKET", flush=True)

    rgb_window = "Aria RGB"
    cv2.namedWindow(rgb_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(rgb_window, 1024, 1024)
    cv2.moveWindow(rgb_window, 50, 50)

    # Main thread is sending images.
    # Create a TCP/IP client socket and establish a connection to the server.
    # The socket uses IPv4 (AF_INET) and TCP (SOCK_STREAM) for data transmission.
    print("START MainThread...", flush=True)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as conn:
        
        print("CONNECT TO SERVER..", flush=True)
        conn.connect((SERVER_HOST, SERVER_PORT))

        # Second thread is recieving files
        print("START RecvThread...", flush=True)
        recv_obj = threading.Thread(target=recv_thread, args=(conn, stop_event), name="RecvThread")
        recv_obj.start()

        print("SOCKET IS READY FOR TRANSMISSION", flush=True)

        # gcount = 0
        while not quit_keypress():
            if aria.CameraId.Rgb in observer.images:

                # Health state of a camera
                # gcount = 0
                
                # Capture the image from the camera
                rgb_image = np.rot90(observer.images[aria.CameraId.Rgb], -1)
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                
                # Get current time in milliseconds rounded to nearest int
                curr_ts = np.array([1000 * time.time()], dtype="i8")

                # Send the image from the camera to Jetson server
                send_array(conn, curr_ts)
                send_array(conn, rgb_image)
                print(f"[{threading.current_thread().name}] Client sent (time, image) to server", flush=True)
                most_recent_buff.append((curr_ts, rgb_image))
                sent_count += 1

                del observer.images[aria.CameraId.Rgb]

                # Visualize if we have received bounding boxes, confidences, and class IDs
                if len(most_recent_buff) > 0 and len(most_recent_bbox) > 0:
                    
                    # Take results from server
                    # recv_ts is a timestamp of the processed image (which we will find in the buffer)
                    recv_ts, recv_boxes, recv_confidences, recv_class_ids = most_recent_bbox.pop()

                    # Find the image on client side by using recieved timestamp from server
                    img = -1
                    for (tic, img) in most_recent_buff:
                        if tic == recv_ts:
                            break
                        if tic  > recv_ts:
                            raise ValueError("Wrong timestamp")
                    
                    # Visualisation
                    img_with_boxes = draw_labels_and_boxes(
                        img=img.copy(), 
                        boxes=recv_boxes, 
                        confidences=recv_confidences, 
                        class_ids=recv_class_ids, 
                        classes=classes
                    )
                    cv2.imshow(rgb_window, img_with_boxes)
            # else:
            #     time.sleep(0.1)
            #     gcount += 1
            
            # if (gcount > 30):  # --> possibly broken camera stream, should interrupt?
            #     print("gcount overflow detected")
            #     break

        # Send termination signal to server
        stop_event.set()
        send_array(conn, None) # tic
        send_array(conn, None) # arr
        print(f"[{threading.current_thread().name}] Client sent termination signal. Disconected from server.", flush=True)
        recv_obj.join()

        print()
        print(f"[{threading.current_thread().name}] Statistics", flush=True)
        print(f"[{threading.current_thread().name}] sent_count={sent_count}", flush=True)
        print(f"[{threading.current_thread().name}] recv_count={recv_count}", flush=True)

        print("Stop listening to image data", flush=True)
        streaming_client.unsubscribe()
        streaming_manager.stop_streaming()
        device_client.disconnect(device)
        
    cv2.destroyAllWindows()


if __name__ == "__main__":
    np.random.seed(0)

    args = parse_args()

    classes = []
    with open("./yolo_models/coco.names.txt", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    run_client(classes=classes)
