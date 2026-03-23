import cv2
from ultralytics import YOLO
import threading
import os

# ===== 路径 =====
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'runs', 'detect', 'train7', 'weights', 'best.pt')

# ===== 模型 =====
model = YOLO(MODEL_PATH)

# ===== 全局变量 =====
latest_result = 0
running = False
thread_started = False
cap = None


# ===== 后台线程 =====
def camera_loop():
    global latest_result, running, cap, thread_started

    # 👉 每次都重新创建摄像头
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera not opened")
        thread_started = False
        return

    cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)

    while running:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("YOLO Detection", annotated_frame)

        key = cv2.waitKey(1) & 0xFF

        # ESC / q 退出
        if key == 27 or key == ord('q'):
            print("Closing camera...")
            running = False
            break

        # ===== 分类 =====
        if len(results[0].boxes) == 0:
            latest_result = 0
            continue

        cls = int(results[0].boxes.cls[0])
        label = results[0].names[cls]

        if label == 'apple':
            latest_result = 1
        elif label == 'banana':
            latest_result = 2
        elif label == 'orange':
            latest_result = 3
        elif label == 'strawberry':
            latest_result = 4
        else:
            latest_result = 0

    # ===== 完整释放 =====
    if cap is not None:
        cap.release()

    cv2.destroyAllWindows()

    # 👉 重置状态（关键！！！）
    thread_started = False
    running = False

    print("Camera fully released.")


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