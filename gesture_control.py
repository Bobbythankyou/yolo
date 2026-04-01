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
STEP_XY = 0.02

# ===== Z控制 =====
Z_DEADZONE = 0.01
Z_SCALE = 5.0
Z_STEP = 0.02

OPEN_THRESHOLD = 3
CLOSE_THRESHOLD = 3

MIN_BOUND = np.array([0.4, -0.4, 0.15])
MAX_BOUND = np.array([0.9,  0.4, 0.6])

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

                # =========================
                # ===== XY 单轴控制 =====
                # =========================
                dx = cx - center[0]
                dy = cy - center[1]
                dist = np.sqrt(dx**2 + dy**2)

                move_xy = np.array([0.0, 0.0, 0.0])

                if dist > RADIUS:
                    # 👉 判断主方向（只允许一个轴动）
                    if abs(dx) > abs(dy):
                        # 左右
                        if dx > 0:
                            move_xy[0] = STEP_XY
                        else:
                            move_xy[0] = -STEP_XY
                    else:
                        # 上下
                        if dy > 0:
                            move_xy[1] = STEP_XY
                        else:
                            move_xy[1] = -STEP_XY

                # =========================
                # ===== Z 滑杆控制 =====
                # =========================
                hand_size = get_hand_size(handLms)

                if base_hand_size is None:
                    base_hand_size = hand_size

                dz = hand_size - base_hand_size

                if abs(dz) < Z_DEADZONE:
                    dz = 0
                else:
                    dz = dz * Z_SCALE

                dz = np.clip(dz, -1, 1)

                move_z = np.array([0.0, 0.0, dz * Z_STEP])

                # ===== 合并 =====
                current_target += move_xy + move_z

                # ===== 软边界 =====
                for i in range(3):
                    if current_target[i] < MIN_BOUND[i]:
                        current_target[i] += 0.003
                    elif current_target[i] > MAX_BOUND[i]:
                        current_target[i] -= 0.003

                # ===== 更新基准 =====
                base_hand_size = 0.95 * base_hand_size + 0.05 * hand_size

        # ===== UI =====
        cv2.circle(frame, center, RADIUS, (0, 255, 0), 2)

        cv2.putText(frame, f"XYZ: {np.round(current_target,3)}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.putText(frame, f"Gripper: {gripper}",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)

        # ===== Z进度条 =====
        z_norm = (current_target[2] - MIN_BOUND[2]) / (MAX_BOUND[2] - MIN_BOUND[2])
        bar_x = int(20 + z_norm * 200)

        cv2.rectangle(frame, (20, 100), (220, 120), (100, 100, 100), 2)
        cv2.rectangle(frame, (20, 100), (bar_x, 120), (0, 255, 255), -1)

        cv2.putText(frame, "Z", (230, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv2.imshow("Gesture Control", frame)

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