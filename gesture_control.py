import cv2
import mediapipe as mp
import numpy as np

# ===== 全局控制变量 =====
current_target = np.array([0.65, 0.0, 0.29])
gripper = 0

# ===== 参数 =====
RADIUS = 150
STEP_XY = 0.01
SMOOTH = 0.2

# ===== Z轴增强参数（🔥重点）=====
Z_BASE_GAIN = 0.05
Z_DYNAMIC_GAIN = 0.3

# ===== 手势稳定参数 =====
OPEN_THRESHOLD = 3
CLOSE_THRESHOLD = 3

open_counter = 0
close_counter = 0

# ===== Z轴基准 =====
base_hand_size = None

# ===== 工作空间限制 =====
MIN_BOUND = np.array([0.5, -0.3, 0.2])
MAX_BOUND = np.array([0.8,  0.3, 0.5])

# ===== Mediapipe =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)


# ===== 判断手是否张开 =====
def is_hand_open(hand_landmarks):
    tips = [8, 12, 16, 20]
    open_count = 0

    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            open_count += 1

    return open_count >= 2   # 放宽阈值


# ===== 手掌大小（用于Z）=====
def get_hand_size(handLms):
    x1 = handLms.landmark[5].x
    y1 = handLms.landmark[5].y
    x2 = handLms.landmark[17].x
    y2 = handLms.landmark[17].y
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


# ===== 主循环 =====
def run_gesture_control():
    global current_target, gripper
    global open_counter, close_counter, base_hand_size

    print("[INIT] Gesture control (Enhanced Z-axis)")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        center = (w // 2, h // 2)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # ===== 画安全圆 =====
        cv2.circle(frame, center, RADIUS, (0, 255, 0), 2)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

                # ===== 手掌中心 =====
                cx = int(handLms.landmark[0].x * w)
                cy = int(handLms.landmark[0].y * h)
                cv2.circle(frame, (cx, cy), 10, (255, 0, 0), -1)

                dist = np.sqrt((cx - center[0])**2 + (cy - center[1])**2)

                # =========================
                # ✅ 1. 夹爪控制（稳定）
                # =========================
                if is_hand_open(handLms):
                    open_counter += 1
                    close_counter = 0
                else:
                    close_counter += 1
                    open_counter = 0

                if gripper == 1:
                    if open_counter >= OPEN_THRESHOLD:
                        gripper = 0
                        print("[GRIPPER] OPEN")
                else:
                    if close_counter >= CLOSE_THRESHOLD:
                        gripper = 1
                        print("[GRIPPER] CLOSE")

                # =========================
                # ✅ 2. XY控制（圆外才动）
                # =========================
                if dist >= RADIUS:
                    dx = (cx - center[0]) / w
                    dy = (cy - center[1]) / h

                    target_xy = np.array([
                        dx * STEP_XY,
                        dy * STEP_XY,
                        0
                    ])

                    current_target = (1 - SMOOTH) * current_target + SMOOTH * (current_target + target_xy)

                else:
                    cv2.putText(frame, "SAFE", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # =========================
                # 🔥 3. Z轴增强控制（核心）
                # =========================
                hand_size = get_hand_size(handLms)

                if base_hand_size is None:
                    base_hand_size = hand_size

                dz = hand_size - base_hand_size

                # ===== 限制异常跳变 =====
                dz = np.clip(dz, -0.05, 0.05)

                # ===== 动态增益（关键🔥）=====
                gain = Z_BASE_GAIN + Z_DYNAMIC_GAIN * abs(dz)

                current_target[2] += dz * gain

                # ===== 更新基准（缓慢）=====
                base_hand_size = 0.9 * base_hand_size + 0.1 * hand_size

                # =========================
                # 限制工作空间
                # =========================
                current_target[:] = np.clip(current_target, MIN_BOUND, MAX_BOUND)

                # =========================
                # 显示
                # =========================
                cv2.putText(frame, f"XYZ: {np.round(current_target, 3)}",
                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                cv2.putText(frame, f"Gripper: {gripper}",
                            (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

                cv2.putText(frame, f"dz: {round(dz,4)}",
                            (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Gesture Control (Final)", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            print("[EXIT]")
            break

    cap.release()
    cv2.destroyAllWindows()


# ===== Simulink接口 =====
def get_control():
    return [
        float(current_target[0]),
        float(current_target[1]),
        float(current_target[2]),
        float(gripper)
    ]


if __name__ == "__main__":
    run_gesture_control()