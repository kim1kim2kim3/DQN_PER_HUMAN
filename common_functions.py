# common_functions.py
import cv2
import numpy as np

def process_state_image(state):
    # BGR 이미지를 그레이스케일로 변환 후 정규화
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = state.astype(float)
    state /= 255.0
    return state

def generate_state_frame_stack_from_queue(deque):
    frame_stack = np.array(deque)
    # (stack, x, y) → (x, y, stack) 으로 차원 재배열
    return np.transpose(frame_stack, (1, 2, 0))
