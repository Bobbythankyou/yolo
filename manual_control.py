import numpy as np
import threading
import keyboard
import time

# ===== 全局变量 =====
current_target = np.array([0.65, 0.0, 0.29], dtype=float)
gripper = 0

step = 0.01

thread_started = False
debug_started = False
running = True


# ===== 键盘回调 =====
def on_key(event):
    global current_target, gripper, running

    key = event.name
    print(f"[KEY] Pressed: {key}")

    # ===== 平面 + 高度控制 =====
    # x轴：前后
    if key == 'i':
        current_target[0] += step   # 前
    elif key == 'k':
        current_target[0] -= step   # 后

    # y轴：左右
    elif key == 'a':
        current_target[1] += step   # 左
    elif key == 'd':
        current_target[1] -= step   # 右

    # z轴：上下
    elif key == 'w':
        current_target[2] += step   # 上
    elif key == 's':
        current_target[2] -= step   # 下

    # ===== 抓手 =====
    elif key == 'q':
        gripper = 1   # 抓取
    elif key == 'e':
        gripper = 0   # 松开

    # ===== 退出 =====
    elif key == 'esc':
        print("[EXIT] ESC pressed, shutting down...")
        running = False
        keyboard.unhook_all()

    print(f"[STATE] target={current_target}, gripper={gripper}")


# ===== 启动键盘监听 =====
def start_keyboard():
    global thread_started
    if not thread_started:
        print("[INIT] Keyboard listener started ✅")
        keyboard.on_press(on_key)
        thread_started = True


# ===== debug线程 =====
def debug_loop():
    global running
    while running:
        print(f"[LOOP] current_target={current_target}, gripper={gripper}")
        time.sleep(2)
    print("[DEBUG] loop stopped")


def start_debug():
    global debug_started
    if not debug_started:
        t = threading.Thread(target=debug_loop, daemon=True)
        t.start()
        debug_started = True


# ===== 主接口 =====
def get_control():
    global running

    if not running:
        print("[INFO] System stopped")
        return [0.0, 0.0, 0.0, 0.0]

    start_keyboard()
    start_debug()

    print("[CALL] get_control called")

    return [
        float(current_target[0]),
        float(current_target[1]),
        float(current_target[2]),
        float(gripper)
    ]