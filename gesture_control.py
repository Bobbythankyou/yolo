import cv2
import mediapipe as mp
import numpy as np
import threading
import time

# ===== 全局变量 =====
current_target = np.array([0.65, 0.0, 0.29])
gripper = 0

running = True
thread_started = False

# ===== 参数 =====
RADIUS = 150
STEP_XY = 0.015   # 稍微调大一点更自然

Z_BASE_GAIN = 0.05
Z_DYNAMIC_GAIN = 0.3

OPEN_THRESHOLD = 3
CLOSE_THRESHOLD = 3

MIN_BOUND = np.array([0.5, -0.3, 0.2])
MAX_BOUND = np.array([0.8,  0.3, 0.5])

# ===== Mediapipe =====
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# ===== 手势判断 =====
def is_hand_open(hand_landmarks):
    tips = [8, 12, 16, 20]
    open_count = 0
    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            open_count += 1
    return open_count >= 2


def get_hand_size(handLms):
    x1 = handLms.landmark[5].x
    y1 = handLms.landmark[5].y
    x2 = handLms.landmark[17].x
    y2 = handLms.landmark[17].y
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


# ===== 摄像头线程 =====
def camera_loop():
    global current_target, gripper, running, thread_started

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("[ERROR] Camera failed")
        thread_started = False
        return

    hands = mp_hands.Hands(max_num_hands=1)

    open_counter = 0
    close_counter = 0
    base_hand_size = None

    print("[INIT] Gesture thread started")

    while running:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        center = (w // 2, h // 2)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:

                # ===== 画手 =====
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

                # ===== 手掌中心 =====
                cx = int(handLms.landmark[0].x * w)
                cy = int(handLms.landmark[0].y * h)

                # ===== Gripper =====
                if is_hand_open(handLms):
                    open_counter += 1
                    close_counter = 0
                else:
                    close_counter += 1
                    open_counter = 0

                if gripper == 1:
                    if open_counter >= OPEN_THRESHOLD:
                        gripper = 0
                else:
                    if close_counter >= CLOSE_THRESHOLD:
                        gripper = 1

                # ===== 统一3D控制（核心🔥）=====

                # XY方向（相对中心）
                dx = (cx - center[0]) / w
                dy = (cy - center[1]) / h

                # 👉 死区（避免抖动）
                if abs(dx) < 0.05:
                    dx = 0
                if abs(dy) < 0.05:
                    dy = 0

                # Z方向（手掌大小）
                hand_size = get_hand_size(handLms)

                if base_hand_size is None:
                    base_hand_size = hand_size

                dz = hand_size - base_hand_size
                dz = np.clip(dz, -0.05, 0.05)

                # 平滑更新
                base_hand_size = 0.9 * base_hand_size + 0.1 * hand_size

                # ===== 合成3D向量 =====
                move_vec = np.array([
                    dx * STEP_XY,
                    dy * STEP_XY,
                    dz * (Z_BASE_GAIN + Z_DYNAMIC_GAIN * abs(dz))
                ])

                # ===== 一次性更新（关键🔥）=====
                current_target += move_vec

                # ===== 限制范围 =====
                current_target[:] = np.clip(current_target, MIN_BOUND, MAX_BOUND)

        # ===== 可视化 =====
        cv2.circle(frame, center, RADIUS, (0, 255, 0), 2)

        cv2.putText(frame, f"XYZ: {np.round(current_target,3)}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.putText(frame, f"Gripper: {gripper}",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)

        cv2.imshow("Gesture Control", frame)

        # ESC退出
        if cv2.waitKey(1) & 0xFF == 27:
            break

        time.sleep(0.01)

    print("[STOP] Camera stopped")

    cap.release()
    cv2.destroyAllWindows()
    thread_started = False


# ===== 启动线程 =====
def start_thread():
    global thread_started

    if not thread_started:
        t = threading.Thread(target=camera_loop, daemon=True)
        t.start()
        thread_started = True


# ===== Simulink接口 =====
def get_control():
    start_thread()

    return [
        float(current_target[0]),
        float(current_target[1]),
        float(current_target[2]),
        float(gripper)
    ]