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
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional, List, Dict, Any
import uvicorn
from contextlib import asynccontextmanager
import io

# ========== GLOBAL VARIABLES ==========
# WebSocket connection
websocket_client: Optional['YOLOWebSocketClient'] = None
detection_active = False
current_frame = None
current_annotated_frame = None  # Frame với detection boxes
detection_results = []
current_fps = 0.0
client_thread = None

# Device setup
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🔧 Sử dụng device: {device}")

# YOLO Model
yolo_model = YOLO('main_model_v2.pt')

# FPS Control
TARGET_FPS = 5
FRAME_INTERVAL = 1.0 / TARGET_FPS  # 0.2 seconds

def get_local_ip():
    """Lấy IP address chính xác của máy tính"""
    try:
        # Tạo socket để connect ra ngoài (không thực sự connect)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
        return local_ip
    except Exception:
        # Fallback method
        hostname = socket.gethostname()
        return socket.gethostbyname(hostname)

def get_all_ip_addresses():
    """Lấy tất cả IP addresses có thể dùng"""
    import socket
    hostname = socket.gethostname()
    ip_list = []
    
    try:
        # Primary IP
        primary_ip = get_local_ip()
        ip_list.append(primary_ip)
        
        # Additional IPs
        for info in socket.getaddrinfo(hostname, None):
            ip = info[4][0]
            if ip not in ip_list and not ip.startswith('127.') and ':' not in ip:
                ip_list.append(ip)
                
    except Exception as e:
        print(f"⚠️ Lỗi lấy IP: {e}")
        ip_list = ["localhost"]
    
    return ip_list

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
            if not frame_base64:
                return None
                
            frame_bytes = base64.b64decode(frame_base64)
            if len(frame_bytes) == 0:
                print("⚠️ Frame bytes rỗng")
                return None
                
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if frame is None:
                print("⚠️ cv2.imdecode trả về None")
                return None
                
            return frame
        except Exception as e:
            print(f"❌ Lỗi decode frame: {e}")
            return None
    
    def process_frame_yolo(self, frame):
        """Xử lý frame bằng YOLO và trả về kết quả"""
        global detection_results, current_annotated_frame
        try:
            # Kiểm tra frame hợp lệ
            if frame is None or frame.size == 0:
                print("⚠️ Frame không hợp lệ")
                return frame, []
            
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
            
            # Create annotated frame - với error handling
            try:
                annotated_frame = results[0].plot()
                current_annotated_frame = annotated_frame.copy()
            except Exception as plot_error:
                print(f"⚠️ Lỗi plot results: {plot_error}")
                annotated_frame = frame.copy()
                current_annotated_frame = frame.copy()
            
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
        """Main client loop với FPS control - KHÔNG DÙNG cv2.imshow()"""
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
                                    
                                    # Log detection results thay vì hiển thị
                                    if objects:
                                        print(f"🎯 Detected: {', '.join(objects)}")
                                    
                                    # Tính FPS
                                    self.frame_count += 1
                                    elapsed_time = current_time - self.start_time
                                    if elapsed_time >= 1.0:
                                        current_fps = self.frame_count / elapsed_time
                                        self.frame_count = 0
                                        self.start_time = current_time
                                        print(f"📊 Detection FPS: {current_fps:.2f}")
                                
                                self.last_frame_time = current_time
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
        
        # Cleanup - KHÔNG CẦN cv2.destroyAllWindows()
        if self.websocket:
            await self.websocket.close()
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
    SERVER_IP = "172.20.10.3"  # IP của Raspberry Pi
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

# ========== WEB STREAMING ENDPOINTS ==========

@app.get("/", response_class=HTMLResponse)
async def get_web_viewer():
    """Trang web để xem video stream"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>YOLO Detection Stream</title>
        <meta charset="utf-8">
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background: #f0f0f0;
            }
            .container { 
                max-width: 1200px; 
                margin: 0 auto; 
                background: white; 
                padding: 20px; 
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .header { 
                text-align: center; 
                margin-bottom: 20px; 
            }
            .video-container { 
                text-align: center; 
                margin: 20px 0; 
            }
            .video-stream { 
                max-width: 100%; 
                border: 2px solid #333; 
                border-radius: 10px;
            }
            .controls { 
                text-align: center; 
                margin: 20px 0; 
            }
            .btn { 
                padding: 10px 20px; 
                margin: 5px; 
                border: none; 
                border-radius: 5px; 
                cursor: pointer; 
                font-size: 16px;
            }
            .btn-start { background: #4CAF50; color: white; }
            .btn-stop { background: #f44336; color: white; }
            .btn-refresh { background: #2196F3; color: white; }
            .status { 
                margin: 20px 0; 
                padding: 15px; 
                border-radius: 5px;
                background: #e8f5e8;
                border: 1px solid #4CAF50;
            }
            .objects { 
                margin: 10px 0; 
                padding: 10px; 
                background: #f9f9f9; 
                border-radius: 5px;
            }
            .object-tag { 
                display: inline-block; 
                background: #2196F3; 
                color: white; 
                padding: 5px 10px; 
                margin: 2px; 
                border-radius: 15px; 
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎯 YOLO Detection Video Stream</h1>
                <p>Real-time object detection từ Raspberry Pi camera</p>
            </div>
            
            <div class="video-container">
                <img id="videoStream" class="video-stream" src="/video_feed" alt="Video Stream">
            </div>
            
            <div class="controls">
                <button class="btn btn-start" onclick="startDetection()">🎯 Bắt đầu Detection</button>
                <button class="btn btn-stop" onclick="stopDetection()">🛑 Dừng Detection</button>
                <button class="btn btn-refresh" onclick="refreshStatus()">🔄 Refresh Status</button>
            </div>
            
            <div id="status" class="status">
                <h3>📊 Trạng thái:</h3>
                <div id="statusContent">Loading...</div>
            </div>
            
            <div class="objects">
                <h3>🎯 Detected Objects:</h3>
                <div id="detectedObjects">No objects detected</div>
            </div>
        </div>

        <script>
            let detectionActive = false;
            
            async function startDetection() {
                try {
                    const response = await fetch('/detect', { method: 'POST' });
                    const data = await response.json();
                    detectionActive = true;
                    updateStatus();
                    alert('✅ Detection started!');
                } catch (error) {
                    alert('❌ Error starting detection: ' + error);
                }
            }
            
            async function stopDetection() {
                try {
                    const response = await fetch('/stop', { method: 'POST' });
                    const data = await response.json();
                    detectionActive = false;
                    updateStatus();
                    alert('🛑 Detection stopped!');
                } catch (error) {
                    alert('❌ Error stopping detection: ' + error);
                }
            }
            
            async function updateStatus() {
                try {
                    const response = await fetch('/status');
                    const data = await response.json();
                    
                    document.getElementById('statusContent').innerHTML = `
                        <p><strong>Active:</strong> ${data.active ? '✅ Yes' : '❌ No'}</p>
                        <p><strong>FPS:</strong> ${data.fps}</p>
                        <p><strong>Device:</strong> ${data.device}</p>
                        <p><strong>Connection:</strong> ${data.connection_stable ? '✅ Stable' : '❌ Unstable'}</p>
                    `;
                    
                    const objectsDiv = document.getElementById('detectedObjects');
                    if (data.objects && data.objects.length > 0) {
                        objectsDiv.innerHTML = data.objects.map(obj => 
                            `<span class="object-tag">${obj}</span>`
                        ).join('');
                    } else {
                        objectsDiv.innerHTML = 'No objects detected';
                    }
                } catch (error) {
                    console.error('Error updating status:', error);
                }
            }
            
            function refreshStatus() {
                updateStatus();
                // Refresh video stream
                const img = document.getElementById('videoStream');
                img.src = img.src.split('?')[0] + '?t=' + new Date().getTime();
            }
            
            // Auto-refresh status every 2 seconds
            setInterval(updateStatus, 2000);
            
            // Initial status load
            updateStatus();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

def generate_frames():
    """Generator để stream video frames"""
    global current_frame, current_annotated_frame, detection_active
    
    while True:
        try:
            # Chọn frame để stream
            if detection_active and current_annotated_frame is not None:
                frame = current_annotated_frame.copy()
            elif current_frame is not None:
                frame = current_frame.copy()
            else:
                # Tạo frame placeholder
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, 'No Video Feed', (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.1)  # Limit streaming FPS
            
        except Exception as e:
            print(f"❌ Lỗi generate frame: {e}")
            time.sleep(0.5)

@app.get("/video_feed")
async def video_feed():
    """MJPEG video stream endpoint"""
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# ========== EXISTING API ENDPOINTS ==========
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
    print(f"\n{'='*60}")
    print(f"🎯 YOLO DETECTION CLIENT & WEB STREAMING API")
    print(f"{'='*60}")
    print(f"🔧 PyTorch: {torch.__version__}")
    print(f"🔧 Device: {device}")
    print(f"🔧 Target FPS: {TARGET_FPS}")
    print(f"🔧 YOLO Model: main_model_v2.pt")
    
    # Lấy IP addresses
    primary_ip = get_local_ip()
    all_ips = get_all_ip_addresses()
    
    print(f"\n{'🌐 NETWORK INFORMATION':^60}")
    print(f"{'='*60}")
    print(f"📡 Primary IP: {primary_ip}")
    
    if len(all_ips) > 1:
        print(f"📡 Alternative IPs:")
        for ip in all_ips[1:]:
            print(f"   • {ip}")
    
    print(f"\n{'🌐 WEB VIDEO STREAMING':^60}")
    print(f"{'='*60}")
    print(f"🔗 Web Viewer: http://{primary_ip}:8000")
    print(f"📺 Video Stream: http://{primary_ip}:8000/video_feed")
    print(f"🎮 Mở trình duyệt và truy cập link trên để xem video!")
    
    print(f"\n{'📱 API ENDPOINTS':^60}")
    print(f"{'='*60}")
    print(f"📋 API Endpoints:")
    print(f"   • POST /detect  - Bắt đầu detection")
    print(f"   • POST /stop    - Dừng detection") 
    print(f"   • GET /status   - Xem trạng thái")
    print(f"   • GET /test     - Test connection")
    print(f"   • GET /frame    - Lấy frame hiện tại")
    
    print(f"\n{'🤖 RASPBERRY PI CONNECTION':^60}")
    print(f"{'='*60}")
    print(f"🔗 WebSocket sẽ kết nối tới: ws://172.20.10.3:8765")
    print(f"📷 Nhận video stream từ PiCamera2")
    
    print(f"\n{'🚀 STARTING SERVER':^60}")
    print(f"{'='*60}")
    print(f"⏳ Khởi động Web Streaming Server trên cổng 8000...")
    print(f"📝 Nhấn Ctrl+C để dừng server")
    print(f"{'='*60}\n")
    
    # Chạy FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
