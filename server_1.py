from fastapi import FastAPI, HTTPException
import cv2
import torch
from ultralytics import YOLO
import time
import socket
import threading
import asyncio
from typing import Optional

app = FastAPI()

# ========== MPS SETUP ==========
def setup_device():
    if torch.backends.mps.is_available():
        device = "mps"
        print("✅ Sử dụng MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = "cuda" 
        print("✅ Sử dụng CUDA GPU")
    else:
        device = "cpu"
        print("⚠️ Sử dụng CPU")
    return device

device = setup_device()

# Load model với MPS
model = YOLO('main_model_v2.pt')
model.to(device)
print(f"Model device: {model.device}")

# ========== GLOBAL VARIABLES ==========
cap: Optional[cv2.VideoCapture] = None
detection_active = False
detection_results = []
current_fps = 0.0
detection_thread = None

# Target FPS cho detection (2 FPS)
TARGET_FPS = 2
FRAME_INTERVAL = 1.0 / TARGET_FPS  # 0.5 giây mỗi frame

# ========== MAIN DETECTION FUNCTION ==========
def detection_worker():
    global cap, detection_active, detection_results, current_fps
    
    # Khởi tạo camera
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ Không thể mở camera")
        detection_active = False
        return
    
    # Tối ưu camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # FPS tracking
    detection_count = 0
    start_time = time.time()
    last_detection_time = 0
    
    print(f"📹 Camera đã sẵn sàng với {TARGET_FPS} FPS detection.")
    
    try:
        while detection_active:
            current_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Chỉ detect theo interval đã định (2 FPS)
            if current_time - last_detection_time >= FRAME_INTERVAL:
                try:
                    # YOLO detection với MPS
                    results = model(frame, conf=0.5, device=device, verbose=False)
                    
                    # Xử lý kết quả - CHỈ LẤY TÊN OBJECT
                    detected_objects = []
                    if results[0].boxes is not None:
                        for box in results[0].boxes:
                            cls = int(box.cls[0])
                            label = model.names[cls]
                            # Chỉ thêm tên, không thêm confidence
                            if label not in detected_objects:  # Tránh duplicate
                                detected_objects.append(label)
                    
                    detection_results = detected_objects
                    last_detection_time = current_time
                    
                    # Tính FPS thực tế
                    detection_count += 1
                    elapsed = current_time - start_time
                    if elapsed >= 1.0:
                        current_fps = detection_count / elapsed
                        detection_count = 0
                        start_time = current_time
                    
                except Exception as e:
                    print(f"Lỗi detection: {e}")
                    continue
            
            # Hiển thị frame (với FPS cao hơn để smooth)
            try:
                annotated_frame = frame.copy()
                
                # Vẽ detection results lên frame
                if detection_results:
                    y_offset = 60
                    for obj in detection_results:
                        cv2.putText(annotated_frame, obj, (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        y_offset += 30
                
                cv2.putText(annotated_frame, f'Detection FPS: {current_fps:.1f} | Device: {device}', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Sign Language Detection', annotated_frame)
                
                # Thoát khi nhấn 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("🛑 Dừng detection từ keyboard")
                    detection_active = False
                    break
                    
            except Exception as e:
                print(f"Lỗi hiển thị: {e}")
                continue
    
    except Exception as e:
        print(f"Lỗi camera worker: {e}")
    
    finally:
        # Cleanup - ĐẢM BẢO CLEANUP HOÀN TOÀN
        print("🧹 Cleaning up camera resources...")
        detection_active = False
        
        if cap:
            cap.release()
            cap = None
        
        cv2.destroyAllWindows()
        
        # Đợi một chút để đảm bảo resources được giải phóng
        time.sleep(0.5)
        print("✅ Cleanup hoàn tất")

# ========== API ENDPOINTS ==========
@app.post("/detect")
async def start_detection():
    """Bắt đầu detection"""
    global detection_active, detection_thread
    
    try:
        if not detection_active:
            detection_active = True
            detection_thread = threading.Thread(target=detection_worker)
            detection_thread.daemon = True
            detection_thread.start()
            
            await asyncio.sleep(1.0)  # Đợi khởi tạo lâu hơn
        
        return {
            "status": "detecting",
            "objects": detection_results,
            "fps": round(current_fps, 1),
            "device": device,
            "target_fps": TARGET_FPS
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop")
async def stop_detection():
    """Dừng detection - ĐẢM BẢO STOP HOÀN TOÀN"""
    global detection_active, cap, detection_thread
    
    print("🛑 Nhận yêu cầu stop detection...")
    
    # Set flag để dừng loop
    detection_active = False
    
    # Đợi thread kết thúc
    if detection_thread and detection_thread.is_alive():
        print("⏳ Đang đợi detection thread dừng...")
        detection_thread.join(timeout=5.0)
        
        if detection_thread.is_alive():
            print("⚠️ Thread vẫn chạy sau timeout")
        else:
            print("✅ Detection thread đã dừng")
    
    # Force cleanup nếu cần
    if cap:
        print("🔧 Force cleanup camera...")
        cap.release()
        cap = None
    
    cv2.destroyAllWindows()
    
    # Reset variables
    global detection_results, current_fps
    detection_results = []
    current_fps = 0.0
    
    # Đợi một chút để đảm bảo cleanup
    await asyncio.sleep(0.5)
    
    print("✅ Stop detection hoàn tất")
    
    return {
        "status": "stopped",
        "message": "Detection stopped successfully"
    }

@app.get("/status")
async def get_status():
    """Lấy trạng thái hiện tại"""
    return {
        "active": detection_active,
        "objects": detection_results if detection_active else [],
        "fps": round(current_fps, 1),
        "device": device,
        "target_fps": TARGET_FPS
    }

@app.get("/test")
async def test_connection():
    """Test kết nối"""
    return {
        "status": "connected", 
        "device": device,
        "mps_available": torch.backends.mps.is_available(),
        "target_fps": TARGET_FPS
    }

# ========== SERVER STARTUP ==========
if __name__ == "__main__":
    print(f"\n🚀 Server khởi động...")
    print(f"🔧 PyTorch: {torch.__version__}")
    print(f"🔧 Device: {device}")
    print(f"🔧 Model: {model.device}")
    print(f"🔧 Target Detection FPS: {TARGET_FPS}")
    
    # Lấy IP address
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print(f"\n📡 Server: http://{local_ip}:8000")
    print("📋 Endpoints:")
    print("  POST /detect  - Bắt đầu detection")
    print("  POST /stop    - Dừng detection") 
    print("  GET /status   - Xem trạng thái")
    print("  GET /test     - Test kết nối")
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)