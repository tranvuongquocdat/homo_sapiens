import asyncio
import websockets
import json
import base64
import time
from picamera2 import Picamera2
import cv2
import threading
from queue import Queue

class CameraServer:
    def __init__(self, host="0.0.0.0", port=8765, fps=5):
        self.host = host
        self.port = port
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.frame_queue = Queue(maxsize=2)
        
        # Khởi tạo camera
        self.picam2 = Picamera2()
        # Cấu hình camera với resolution phù hợp
        config = self.picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        
        # Biến để kiểm soát thread
        self.running = False
        self.camera_thread = None
        
    def capture_frames(self):
        """Thread function để capture frame liên tục"""
        last_capture_time = 0
        
        while self.running:
            current_time = time.time()
            
            # Kiểm tra nếu đủ thời gian để capture frame tiếp theo
            if current_time - last_capture_time >= self.frame_interval:
                try:
                    # Capture frame từ camera
                    frame = self.picam2.capture_array()
                    
                    # Convert RGB to BGR cho OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    # Encode frame thành JPEG
                    _, buffer = cv2.imencode('.jpg', frame_bgr, 
                                           [cv2.IMWRITE_JPEG_QUALITY, 80])
                    
                    # Convert sang base64
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Thêm vào queue (replace nếu queue đầy)
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame_base64)
                    else:
                        # Loại bỏ frame cũ và thêm frame mới
                        try:
                            self.frame_queue.get_nowait()
                        except:
                            pass
                        self.frame_queue.put(frame_base64)
                    
                    last_capture_time = current_time
                    
                except Exception as e:
                    print(f"Lỗi capture frame: {e}")
            
            # Sleep ngắn để không tiêu tốn CPU
            time.sleep(0.01)
    
    async def handle_client(self, websocket, path):
        """Xử lý kết nối client"""
        client_address = websocket.remote_address
        print(f"Client connected: {client_address}")
        
        try:
            while True:
                # Chờ request từ client
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    request = json.loads(message)
                    
                    if request.get("action") == "get_frame":
                        # Lấy frame mới nhất từ queue
                        if not self.frame_queue.empty():
                            frame_base64 = self.frame_queue.get()
                            
                            response = {
                                "status": "success",
                                "data": frame_base64,
                                "timestamp": time.time()
                            }
                        else:
                            response = {
                                "status": "no_frame",
                                "message": "Không có frame mới"
                            }
                        
                        await websocket.send(json.dumps(response))
                        
                except asyncio.TimeoutError:
                    # Gửi ping để keep alive
                    try:
                        await websocket.ping()
                    except:
                        break
                except Exception as e:
                    print(f"Lỗi xử lý message: {e}")
                    break
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            print(f"Client disconnected: {client_address}")
    
    async def run_server(self):
        """Async function để chạy server"""
        print(f"Bắt đầu camera capture với FPS: {self.fps}")
        
        # Bắt đầu camera thread
        self.running = True
        self.camera_thread = threading.Thread(target=self.capture_frames)
        self.camera_thread.daemon = True
        self.camera_thread.start()
        
        # Khởi động WebSocket server
        print(f"Khởi động WebSocket server tại ws://{self.host}:{self.port}")
        
        try:
            async with websockets.serve(self.handle_client, self.host, self.port):
                print("Server đang chạy... Nhấn Ctrl+C để dừng")
                await asyncio.Future()  # run forever
        except KeyboardInterrupt:
            print("\nDừng server...")
        finally:
            self.stop_server()
    
    def start_server(self):
        """Khởi động server"""
        try:
            asyncio.run(self.run_server())
        except KeyboardInterrupt:
            print("\nDừng server...")
        finally:
            self.stop_server()
    
    def stop_server(self):
        """Dừng server và giải phóng tài nguyên"""
        self.running = False
        
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2)
        
        if self.picam2:
            self.picam2.stop()
            self.picam2.close()
        
        print("Server đã dừng")

if __name__ == "__main__":
    # Tạo và khởi động server với FPS = 5
    server = CameraServer(host="0.0.0.0", port=8765, fps=5)
    server.start_server()
