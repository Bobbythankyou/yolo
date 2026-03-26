import numpy as np
import threading
import time
import keyboard

# ===== 全局变量 =====
current_target = np.array([0.45, 0.0, 0.20])  # 初始位置
gripper = 0  # 0=open, 1=close

step = 0.01
running = True


# ===== 键盘监听线程 =====
def keyboard_loop():
    global current_target, gripper, running

    while running:
        try:
            # ===== 位置控制 =====
            if keyboard.is_pressed('w'):
                current_target[0] += step
            if keyboard.is_pressed('s'):
                current_target[0] -= step

            if keyboard.is_pressed('a'):
                current_target[1] += step
            if keyboard.is_pressed('d'):
                current_target[1] -= step

            if keyboard.is_pressed('q'):
                current_target[2] += step
            if keyboard.is_pressed('e'):
                current_target[2] -= step

            # ===== 夹爪控制 =====
            if keyboard.is_pressed('o'):
                gripper = 0
            if keyboard.is_pressed('p'):
                gripper = 1

            # ===== 退出 =====
            if keyboard.is_pressed('esc'):
                running = False
                break

            time.sleep(0.05)

        except:
            pass


# ===== 启动线程（只启动一次）=====
thread_started = False

def start_keyboard():
    global thread_started
    if not thread_started:
        t = threading.Thread(target=keyboard_loop, daemon=True)
        t.start()
        thread_started = True


# ===== 给Simulink调用的函数 =====
def get_control():
    start_keyboard()

    return [
        float(current_target[0]),
        float(current_target[1]),
        float(current_target[2]),
        float(gripper)   # ✅ 改这里
    ]