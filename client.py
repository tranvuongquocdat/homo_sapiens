import asyncio
import websockets
import json
import base64
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time

class YOLOClient:
    def __init__(self, server_url, model_path="main_model_v2.pt"):
        self.server_url = server_url
        self.model_path = model_path
        self.model = None
        self.websocket = None
        self.running = False
        self.reconnect_delay = 2  # Delay giữa các lần reconnect
        
        # Load YOLO model
        self.load_model()
        
    def load_model(self):
        """Load YOLO model"""
        try:
            print(f"Đang load model YOLO từ {self.model_path}...")
            self.model = YOLO(self.model_path)
            print("✅ Model đã được load thành công!")
        except Exception as e:
            print(f"⚠️ Lỗi load model: {e}")
            print("🔄 Sử dụng model YOLOv8n mặc định...")
            self.model = YOLO('yolov8n.pt')
    
    def decode_frame(self, frame_base64):
        """Decode frame từ base64"""
        try:
            # Decode base64
            frame_bytes = base64.b64decode(frame_base64)
            
            # Convert bytes to numpy array
            nparr = np.frombuffer(frame_bytes, np.uint8)
            
            # Decode image
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            return frame
        except Exception as e:
            print(f"❌ Lỗi decode frame: {e}")
            return None
    
    def process_frame(self, frame):
        """Xử lý frame bằng YOLO"""
        try:
            # Chạy inference
            results = self.model(frame)
            
            # Vẽ bounding boxes
            annotated_frame = results[0].plot()
            
            return annotated_frame
        except Exception as e:
            print(f"❌ Lỗi xử lý YOLO: {e}")
            return frame
    
    async def request_frame(self):
        """Request frame từ server"""
        try:
            if not self.websocket:
                return None
                
            request = {"action": "get_frame"}
            await self.websocket.send(json.dumps(request))
            
            # Đặt timeout cho response
            response = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
            data = json.loads(response)
            
            if data["status"] == "success":
                return data["data"]
            else:
                return None
                
        except asyncio.TimeoutError:
            print("⏰ Timeout chờ response từ server")
            return None
        except websockets.exceptions.ConnectionClosed:
            print("📡 Kết nối bị đóng")
            return None
        except Exception as e:
            print(f"❌ Lỗi request frame: {e}")
            return None
    
    async def connect_with_retry(self):
        """Kết nối với retry logic"""
        retry_count = 0
        max_retries = 5
        
        while retry_count < max_retries and self.running:
            try:
                print(f"🔄 Đang kết nối tới server: {self.server_url} (lần {retry_count + 1})")
                
                self.websocket = await asyncio.wait_for(
                    websockets.connect(
                        self.server_url,
                        ping_interval=20,
                        ping_timeout=10,
                        close_timeout=10
                    ), 
                    timeout=10.0
                )
                
                print("✅ Kết nối thành công!")
                return True
                
            except Exception as e:
                retry_count += 1
                print(f"❌ Lỗi kết nối (lần {retry_count}): {e}")
                
                if retry_count < max_retries:
                    print(f"⏳ Thử lại sau {self.reconnect_delay} giây...")
                    await asyncio.sleep(self.reconnect_delay)
                    self.reconnect_delay = min(self.reconnect_delay * 1.5, 10)  # Exponential backoff
        
        return False
    
    async def run_client(self):
        """Chạy client chính"""
        self.running = True
        frame_count = 0
        start_time = time.time()
        
        while self.running:
            # Kết nối hoặc reconnect
            if not await self.connect_with_retry():
                print("❌ Không thể kết nối tới server sau nhiều lần thử")
                break
            
            try:
                # Reset reconnect delay khi kết nối thành công
                self.reconnect_delay = 2
                
                while self.running:
                    # Request frame từ server
                    frame_base64 = await self.request_frame()
                    
                    if frame_base64:
                        # Decode frame
                        frame = self.decode_frame(frame_base64)
                        
                        if frame is not None:
                            # Xử lý bằng YOLO
                            processed_frame = self.process_frame(frame)
                            
                            # Hiển thị frame
                            cv2.imshow('YOLO Detection - Nhấn Q để thoát', processed_frame)
                            
                            # Tính FPS
                            frame_count += 1
                            elapsed_time = time.time() - start_time
                            if elapsed_time >= 1.0:
                                fps = frame_count / elapsed_time
                                print(f"📊 FPS: {fps:.2f}")
                                frame_count = 0
                                start_time = time.time()
                            
                            # Kiểm tra phím thoát
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord('q') or key == 27:  # 'q' hoặc ESC
                                self.running = False
                                break
                    else:
                        # Không có frame, server có thể gặp vấn đề
                        await asyncio.sleep(0.1)
                    
                    # Giới hạn FPS client (tối đa 5 FPS)
                    await asyncio.sleep(0.2)  # 200ms = 5 FPS
                    
            except websockets.exceptions.ConnectionClosed:
                print("📡 Kết nối bị đóng bởi server")
                if self.websocket:
                    self.websocket = None
                # Sẽ reconnect trong vòng lặp chính
                
            except Exception as e:
                print(f"❌ Lỗi client: {e}")
                if self.websocket:
                    self.websocket = None
                # Sẽ reconnect trong vòng lặp chính
        
        # Cleanup
        if self.websocket:
            await self.websocket.close()
        cv2.destroyAllWindows()
        print("🛑 Client đã dừng")
    
    def start_client(self):
        """Khởi động client"""
        try:
            asyncio.run(self.run_client())
        except KeyboardInterrupt:
            print("\n🛑 Dừng client...")
            self.running = False

if __name__ == "__main__":
    # Thay đổi IP này thành IP của Raspberry Pi
    SERVER_IP = "192.168.137.19"  # Thay bằng IP thực của Raspberry Pi
    SERVER_PORT = 8765
    SERVER_URL = f"ws://{SERVER_IP}:{SERVER_PORT}"
    
    print("🎯 YOLO Object Detection Client")
    print("📝 Nhấn 'Q' hoặc ESC để thoát")
    print("-" * 50)
    
    client = YOLOClient(SERVER_URL)
    client.start_client()
