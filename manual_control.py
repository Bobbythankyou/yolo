import numpy as np
import threading
import keyboard
import time

current_target = np.array([0.65, 0.0, 0.29])
gripper = 0

step = 0.01
thread_started = False


def on_key(event):
    global current_target, gripper

    key = event.name
    print(f"[KEY] Pressed: {key}")

    if key == 'w':
        current_target[0] += step
    elif key == 's':
        current_target[0] -= step
    elif key == 'a':
        current_target[1] += step
    elif key == 'd':
        current_target[1] -= step
    elif key == 'q':
        current_target[2] += step
    elif key == 'e':
        current_target[2] -= step
    elif key == 'o':
        gripper = 0
    elif key == 'p':
        gripper = 1

    print(f"[STATE] target={current_target}, gripper={gripper}")


def start_keyboard():
    global thread_started

    if not thread_started:
        print("[INIT] Keyboard listener started ✅")

        keyboard.on_press(on_key)

        thread_started = True


# ===== 定时打印（防止你不知道有没有更新）=====
def debug_loop():
    while True:
        print(f"[LOOP] current_target={current_target}, gripper={gripper}")
        time.sleep(2)


def start_debug():
    t = threading.Thread(target=debug_loop, daemon=True)
    t.start()


def get_control():
    start_keyboard()
    start_debug()

    print("[CALL] get_control called")

    return [
        float(current_target[0]),
        float(current_target[1]),
        float(current_target[2]),
        float(gripper)
    ]