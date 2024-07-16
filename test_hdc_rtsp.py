import threading
import time
import numpy as np
import cv2
import re
import os
from PIL import Image

sources = [
    "rtsp://10.32.15.155:18554/CAM01",
    "rtsp://10.32.15.155:18554/CAM02",
    "rtsp://10.32.15.155:18554/CAM03",
]
num_sources = len(sources)
streams = [cv2.VideoCapture(s) for s in sources]
fps = [stream.get(cv2.CAP_PROP_FPS) for stream in streams]
fps = [f if f > 0 else 50 for f in fps]
delays = [int(1000 / f) for f in fps]
frames = [ max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float("inf") 
          for cap in streams ]

imgs = [[] for _ in range(num_sources)]
shape = [[] for _ in range(num_sources)]

vid_stride = 1
buffer = True

def update(i, cap, stream):
    n, f = 0, frames[i]  # frame number, frame array
    while cap.isOpened() and n < (f - 1):
        if len(imgs[i]) < 30:  # keep a <=30-image buffer
            n += 1
            cap.grab()  # .read() = .grab() followed by .retrieve()
            if n % vid_stride == 0:
                success, im = cap.retrieve()
                if not success:
                    im = np.zeros(shape[i], dtype=np.uint8)
                    print("WARNING ⚠️ Video stream unresponsive, please check your IP camera connection.")
                    cap.open(stream)  # re-open stream if signal was lost
                if buffer:
                    imgs[i].append(im)
                else:
                    imgs[i] = [im]
        else:
            time.sleep(0.01)  # wait until the buffer is empty
    

threads = [None] * num_sources

for i, stream in enumerate(streams):
    if not stream.isOpened():
        raise ConnectionError(f"Failed to open")
    w = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sp = (h, w, 3)
    print(f"shape: {sp}")
    shape.append(shape)
    
    threads[i] = threading.Thread(target=update, args=(i, stream, sources[i]), daemon=True)
    threads[i].start()
    

def close():
    for thread in threads:
        if thread.is_alive():
            thread.join(timeout=5)  # Add timeout
    for cap in streams:  # Iterate through the stored VideoCapture objects
        try:
            cap.release()  # release video capture
        except Exception as e:
            print(f"WARNING ⚠️ Could not release VideoCapture object: {e}")
    cv2.destroyAllWindows()

def get_next():
    images = []
    for i, x in enumerate(imgs):
        # Wait until a frame is available in each buffer
        while not x:
            if not threads[i].is_alive() or cv2.waitKey(1) == ord("q"):  # q to quit
                close()
            time.sleep(1 / min(fps))
            x = imgs[i]
            if not x:
                print(f"WARNING ⚠️ Waiting for stream {i}")

        # Get and remove the first frame from imgs buffer
        if buffer:
            images.append(x.pop(0))

        # Get the last frame, and clear the rest from the imgs buffer
        else:
            images.append(x.pop(-1) if x else np.zeros(shape[i], dtype=np.uint8))
            x.clear()
    
    # reshape all images by 640 x 480
    for i, img in enumerate(images):
        images[i] = cv2.resize(img, (640, 480))
    # concat
    images = np.concatenate(images, axis=1)

    return images

while True:
    images = get_next()
    cv2.imshow("Multi-Stream", images)
    if cv2.waitKey(1) == ord("q"):
        break