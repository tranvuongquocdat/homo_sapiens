#!/usr/bin/python3

import math
import sys
import threading
import time
import os
import json
import asyncio
import base64
import cv2
import numpy as np
import websockets
import libcamera
from pathlib import Path
from PIL import Image
from picamera2 import Picamera2
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

# Configure filesystem
script_path = Path(__file__)
script_dir = script_path.parent

# Configure camera
width, height = 640, 480
picam2 = Picamera2()
capture_config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (width, height)}
)
picam2.configure(capture_config)

# Configure interpreter
image_buffer = Image.new("RGB", (width, height))
labels = read_label_file(str(script_dir / "models/mobilenet_coco/coco_labels.txt"))
interpreter = make_interpreter(str(script_dir / "models/mobilenet_coco/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"))
interpreter.allocate_tensors()

detected_objs = []
inference_latency = sys.float_info.max

def is_duplicate(center1, center2):
    dist_thresh = 15
    return dist_thresh >= math.dist(center1, center2)

def run_interpreter():
    global image_buffer, detected_objs, inference_latency

    start = time.perf_counter()

    _, scale = common.set_resized_input(
        interpreter, image_buffer.size, lambda size: image_buffer.resize(size, Image.LANCZOS))
    interpreter.invoke()

    inference_latency = time.perf_counter() - start

    dedup_map = {}
    objs = detect.get_objects(interpreter, 0.5, scale)
    filtered_objs = []

    for obj in objs:
        bbox = obj.bbox
        center = ((bbox.xmax + bbox.xmin) / 2, (bbox.ymax + bbox.ymin) / 2)

        bucket = dedup_map.get(obj.id)
        if bucket is not None:
            should_continue = False
            for other_center in bucket:
                if is_duplicate(center, other_center):
                    should_continue = True
                    break
            if should_continue:
                continue
        else:
            dedup_map[obj.id] = []
        dedup_map[obj.id].append(center)
        filtered_objs.append((obj, bbox))

    detected_objs = filtered_objs

def format_detections():
    detections = []
    for detected_obj in detected_objs:
        obj, bbox = detected_obj
        label = labels.get(obj.id, obj.id)
        # Format: [x1, y1, x2, y2, confidence, class_id, class_name]
        detections.append([
            float(bbox.xmin), float(bbox.ymin), 
            float(bbox.xmax), float(bbox.ymax),
            float(obj.score), int(obj.id), str(label)
        ])
    return detections

def get_cpu_temp():
    temp = os.popen("vcgencmd measure_temp").readline().replace("temp=", "").replace("'C\n", "")
    return float(temp)

async def process_client(websocket):
    global image_buffer, detected_objs
    
    picam2.start()
    
    frame_count = 0
    start_time = time.time()
    fps = 0
    cpu_temp = 0
    
    try:
        while True:
            # Capture image
            pil_image = picam2.capture_image()
            # Convert PIL image to numpy array for OpenCV
            frame = np.array(pil_image)
            
            # Rotate image 90 degrees counter-clockwise
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # Convert to BGR for OpenCV processing (because OpenCV uses BGR)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Convert rotated frame back to PIL for inference
            rotated_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Run inference directly instead of using a thread
            image_buffer.paste(rotated_pil)
            run_interpreter()
            
            # Calculate FPS and CPU temp every 10 frames
            frame_count += 1
            if frame_count % 10 == 0:
                end_time = time.time()
                fps = 10 / (end_time - start_time)
                cpu_temp = get_cpu_temp()
                start_time = time.time()
            
            # Draw bounding boxes for visualization
            frame_with_boxes = frame.copy()
            for detected_obj in detected_objs:
                obj, bbox = detected_obj
                label = labels.get(obj.id, obj.id)
                cv2.rectangle(frame_with_boxes, 
                             (int(bbox.xmin), int(bbox.ymin)), 
                             (int(bbox.xmax), int(bbox.ymax)), 
                             (0, 255, 255), 2)
                cv2.putText(frame_with_boxes, 
                           f"{label}: {obj.score:.2f}", 
                           (int(bbox.xmin), int(bbox.ymin) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Add frame info text - chỉ hiển thị một lần với chữ trắng
            cv2.putText(frame_with_boxes, 
                       f"FPS: {fps:.2f}, Latency: {inference_latency * 1000:.2f}ms, Temp: {cpu_temp:.1f}°C", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Encode image
            _, buffer = cv2.imencode('.jpg', frame_with_boxes)
            img_str = base64.b64encode(buffer).decode('utf-8')
            
            # Format detections
            detections = format_detections()
            
            # Send data
            try:
                await websocket.send(json.dumps({
                    "image": img_str,
                    "detections": detections,
                    "fps": fps,
                    "cpu_temp": cpu_temp,
                    "inference_latency": inference_latency * 1000
                }))
            except websockets.exceptions.ConnectionClosed:
                break
            
            # Small delay to prevent overwhelming the network
            await asyncio.sleep(0.01)
            
    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        picam2.stop()
        print("Waiting for new connection...")

async def main():
    server = await websockets.serve(
        process_client,
        "0.0.0.0",
        8000,
        ping_interval=20,
        ping_timeout=20
    )
    print("Object detection server started on ws://0.0.0.0:8000")
    await server.wait_closed()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer shutting down") 