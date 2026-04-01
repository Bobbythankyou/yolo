import cv2
from ultralytics import YOLO
import threading
import os

# ===== 路径 =====
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'runs', 'detect', 'train7', 'weights', 'best.pt')

# ===== 模型 =====
model = YOLO(MODEL_PATH)

# ===== 参数（核心）=====
CONF_THRESHOLD = 0.75   # ⭐ 你要的阈值

# ===== 全局变量 =====
latest_result = 0
running = False
thread_started = False
cap = None


# ===== 后台线程 =====
def camera_loop():
    global latest_result, running, cap, thread_started

    cap = cv2.VideoCapture(2)

    if not cap.isOpened():
        print("❌ Camera not opened")
        thread_started = False
        return

    cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)

    while running:
        ret, frame = cap.read()
        if not ret:
            continue

        # ===== YOLO 推理 =====
        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("YOLO Detection", annotated_frame)

        key = cv2.waitKey(1) & 0xFF

        # ===== 退出 =====
        if key == 27 or key == ord('q'):
            print("🛑 Closing camera...")
            running = False
            break

        boxes = results[0].boxes

        # ===== 没检测到 =====
        if boxes is None or len(boxes) == 0:
            latest_result = 0
            # print("🚫 No detection")
            continue

        # ===== 取置信度 =====
        confs = boxes.conf.cpu().numpy()
        best_idx = confs.argmax()
        best_conf = confs[best_idx]

        # print(f"🔍 Best confidence: {best_conf:.2f}")

        # ===== 阈值过滤（核心）=====
        if best_conf < CONF_THRESHOLD:
            latest_result = 0
            # print("⚠️ Confidence too low, ignored")
            continue

        # ===== 获取类别 =====
        cls = int(boxes.cls[best_idx])
        label = results[0].names[cls]

        print(f"✅ Detected: {label} ({best_conf:.2f})")

        # ===== 分类映射 =====
        if label == 'strawberry':
            latest_result = 1
        elif label == 'banana':
            latest_result = 2
        elif label == 'Tomato':
            latest_result = 3
        else:
            latest_result = 0

    # ===== 释放资源 =====
    if cap is not None:
        cap.release()

    cv2.destroyAllWindows()

    thread_started = False
    running = False

    print("✅ Camera fully released.")


# ===== 启动线程 =====
def start_camera():
    global thread_started, running

    if not thread_started:
        running = True
        t = threading.Thread(target=camera_loop)
        t.start()
        thread_started = True


# ===== 给 Simulink 调用 =====
def detect_fruit(dummy=0):
    start_camera()
    return latest_result