import asyncio
import websockets
import json
import base64
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import threading
import socket
from fastapi import FastAPI, HTTPException
from typing import Optional, List, Dict, Any
import uvicorn
from contextlib import asynccontextmanager

# ========== GLOBAL VARIABLES ==========
# WebSocket connection
websocket_client: Optional['YOLOWebSocketClient'] = None
detection_active = False
current_frame = None
detection_results = []
current_fps = 0.0
client_thread = None

# Device setup
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🔧 Sử dụng device: {device}")

# YOLO Model
yolo_model = None

# FPS Control
TARGET_FPS = 5
FRAME_INTERVAL = 1.0 / TARGET_FPS  # 0.2 seconds

class YOLOWebSocketClient:
    def __init__(self, server_url, model_path="main_model_v2.pt"):
        self.server_url = server_url
        self.model_path = model_path
        self.model = None
        self.websocket = None
        self.running = False
        self.reconnect_delay = 2
        self.connection_stable = False
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.last_frame_time = 0
        
        # Load YOLO model
        self.load_model()
        
    def load_model(self):
        """Load YOLO model với device optimization"""
        global yolo_model
        try:
            print(f"🔄 Đang load model YOLO từ {self.model_path}...")
            self.model = YOLO(self.model_path)
            self.model.to(device)
            yolo_model = self.model
            print(f"✅ Model loaded successfully on {device}!")
        except Exception as e:
            print(f"⚠️ Lỗi load model: {e}")
            print("🔄 Sử dụng YOLOv8n mặc định...")
            self.model = YOLO('yolov8n.pt')
            self.model.to(device)
            yolo_model = self.model
    
    def decode_frame(self, frame_base64):
        """Decode frame từ base64 với error handling"""
        try:
            frame_bytes = base64.b64decode(frame_base64)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            print(f"❌ Lỗi decode frame: {e}")
            return None
    
    def process_frame_yolo(self, frame):
        """Xử lý frame bằng YOLO và trả về kết quả"""
        global detection_results
        try:
            # YOLO inference
            results = self.model(frame, conf=0.5, device=device, verbose=False)
            
            # Extract detected objects - CHỈ LẤY TÊN OBJECT
            detected_objects = []
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    label = self.model.names[cls]
                    if label not in detected_objects:  # Tránh duplicate
                        detected_objects.append(label)
            
            # Update global detection results
            detection_results = detected_objects
            
            # Create annotated frame
            annotated_frame = results[0].plot()
            
            return annotated_frame, detected_objects
            
        except Exception as e:
            print(f"❌ Lỗi xử lý YOLO: {e}")
            return frame, []
    
    async def request_frame(self):
        """Request frame từ server với timeout optimization"""
        try:
            if not self.websocket:
                return None
                
            request = {"action": "get_frame"}
            await self.websocket.send(json.dumps(request))
            
            # Timeout ngắn hơn để responsive hơn
            response = await asyncio.wait_for(self.websocket.recv(), timeout=3.0)
            data = json.loads(response)
            
            if data.get("status") == "success":
                return data.get("data")
            return None
                
        except asyncio.TimeoutError:
            # Không log quá nhiều timeout để tránh spam
            return None
        except websockets.exceptions.ConnectionClosed:
            print("📡 Kết nối bị đóng")
            self.connection_stable = False
            return None
        except Exception as e:
            print(f"❌ Lỗi request frame: {e}")
            return None
    
    async def connect_with_retry(self):
        """Kết nối với exponential backoff"""
        retry_count = 0
        max_retries = 10
        
        while retry_count < max_retries and self.running:
            try:
                print(f"🔄 Kết nối tới server: {self.server_url} (lần {retry_count + 1})")
                
                self.websocket = await asyncio.wait_for(
                    websockets.connect(
                        self.server_url,
                        ping_interval=15,
                        ping_timeout=8,
                        close_timeout=5,
                        max_size=10**7,  # 10MB để handle large frames
                        read_limit=10**7
                    ), 
                    timeout=8.0
                )
                
                print("✅ Kết nối WebSocket thành công!")
                self.connection_stable = True
                return True
                
            except Exception as e:
                retry_count += 1
                print(f"❌ Lỗi kết nối (lần {retry_count}): {e}")
                
                if retry_count < max_retries:
                    await asyncio.sleep(self.reconnect_delay)
                    self.reconnect_delay = min(self.reconnect_delay * 1.2, 8)  # Exponential backoff
        
        return False
    
    async def run_client(self):
        """Main client loop với FPS control"""
        global detection_active, current_frame, current_fps
        
        self.running = True
        
        while self.running:
            # Kết nối hoặc reconnect
            if not await self.connect_with_retry():
                print("❌ Không thể kết nối tới server")
                break
            
            try:
                self.reconnect_delay = 2  # Reset delay
                
                while self.running and self.connection_stable:
                    current_time = time.time()
                    
                    # FPS Control - chỉ process frame theo interval
                    if current_time - self.last_frame_time >= FRAME_INTERVAL:
                        # Request frame từ server
                        frame_base64 = await self.request_frame()
                        
                        if frame_base64:
                            # Decode frame
                            frame = self.decode_frame(frame_base64)
                            
                            if frame is not None:
                                # Update current frame cho API
                                current_frame = frame.copy()
                                
                                # Xử lý YOLO nếu detection active
                                if detection_active:
                                    processed_frame, objects = self.process_frame_yolo(frame)
                                    
                                    # Hiển thị frame (optional - có thể tắt để tiết kiệm resources)
                                    cv2.imshow('YOLO Detection - Press Q to quit', processed_frame)
                                    
                                    # Tính FPS
                                    self.frame_count += 1
                                    elapsed_time = current_time - self.start_time
                                    if elapsed_time >= 1.0:
                                        current_fps = self.frame_count / elapsed_time
                                        self.frame_count = 0
                                        self.start_time = current_time
                                        print(f"📊 Detection FPS: {current_fps:.2f}")
                                
                                self.last_frame_time = current_time
                                
                                # Check for quit key
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    self.running = False
                                    break
                        else:
                            # Không có frame - có thể server busy
                            await asyncio.sleep(0.1)
                    else:
                        # Chờ để maintain FPS
                        await asyncio.sleep(0.05)
                    
            except websockets.exceptions.ConnectionClosed:
                print("📡 Kết nối bị đóng bởi server")
                self.connection_stable = False
                if self.websocket:
                    self.websocket = None
                    
            except Exception as e:
                print(f"❌ Lỗi client: {e}")
                self.connection_stable = False
                if self.websocket:
                    self.websocket = None
        
        # Cleanup
        if self.websocket:
            await self.websocket.close()
        cv2.destroyAllWindows()
        print("🛑 WebSocket Client đã dừng")
    
    def start_client(self):
        """Khởi động client trong thread riêng"""
        try:
            asyncio.run(self.run_client())
        except KeyboardInterrupt:
            print("\n🛑 Dừng client...")
            self.running = False

# ========== FASTAPI SERVER ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Khởi tạo và cleanup khi start/stop server"""
    global websocket_client, client_thread
    
    # Startup
    print("🚀 Khởi tạo WebSocket Client...")
    SERVER_IP = "192.168.137.19"  # IP của Raspberry Pi
    SERVER_PORT = 8765
    SERVER_URL = f"ws://{SERVER_IP}:{SERVER_PORT}"
    
    websocket_client = YOLOWebSocketClient(SERVER_URL)
    
    # Chạy client trong thread riêng
    client_thread = threading.Thread(target=websocket_client.start_client)
    client_thread.daemon = True
    client_thread.start()
    
    # Đợi client connect
    await asyncio.sleep(2)
    
    yield
    
    # Cleanup
    print("🛑 Đang dừng WebSocket Client...")
    if websocket_client:
        websocket_client.running = False

app = FastAPI(lifespan=lifespan, title="YOLO Detection API", version="1.0.0")

# ========== API ENDPOINTS ==========
@app.post("/detect")
async def start_detection():
    """Bắt đầu YOLO detection"""
    global detection_active
    
    try:
        if not websocket_client or not websocket_client.connection_stable:
            raise HTTPException(status_code=503, detail="WebSocket connection not available")
        
        detection_active = True
        
        # Đợi một frame để có kết quả
        await asyncio.sleep(0.5)
        
        return {
            "status": "detecting",
            "objects": detection_results,
            "fps": round(current_fps, 1),
            "device": device,
            "target_fps": TARGET_FPS,
            "connection_stable": websocket_client.connection_stable
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop")
async def stop_detection():
    """Dừng YOLO detection"""
    global detection_active, detection_results, current_fps
    
    try:
        detection_active = False
        detection_results = []
        current_fps = 0.0
        
        return {
            "status": "stopped",
            "message": "Detection stopped successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    """Lấy trạng thái detection hiện tại"""
    connection_status = websocket_client.connection_stable if websocket_client else False
    
    return {
        "active": detection_active,
        "objects": detection_results if detection_active else [],
        "fps": round(current_fps, 1),
        "device": device,
        "target_fps": TARGET_FPS,
        "connection_stable": connection_status,
        "websocket_url": websocket_client.server_url if websocket_client else None
    }

@app.get("/test")
async def test_connection():
    """Test kết nối và hệ thống"""
    return {
        "status": "connected",
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "target_fps": TARGET_FPS,
        "model_loaded": yolo_model is not None,
        "websocket_stable": websocket_client.connection_stable if websocket_client else False
    }

@app.get("/frame")
async def get_current_frame():
    """Lấy frame hiện tại (base64)"""
    global current_frame
    
    if current_frame is None:
        raise HTTPException(status_code=404, detail="No frame available")
    
    try:
        # Encode frame to base64
        _, buffer = cv2.imencode('.jpg', current_frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "status": "success",
            "frame": frame_base64,
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== MAIN ==========
if __name__ == "__main__":
    print(f"\n🎯 YOLO Detection Client & API Server")
    print(f"🔧 PyTorch: {torch.__version__}")
    print(f"🔧 Device: {device}")
    print(f"🔧 Target FPS: {TARGET_FPS}")
    
    # Lấy IP address
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print(f"\n📡 API Server: http://{local_ip}:8000")
    print("📋 Endpoints cho Mobile App:")
    print("  POST /detect  - Bắt đầu detection")
    print("  POST /stop    - Dừng detection") 
    print("  GET /status   - Xem trạng thái")
    print("  GET /test     - Test connection")
    print("  GET /frame    - Lấy frame hiện tại")
    print("\n🔗 WebSocket sẽ kết nối tới Raspberry Pi...")
    
    # Chạy FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)
