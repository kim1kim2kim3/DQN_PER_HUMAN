# play_car_racing_with_keyboard.py
import gym

# 키보드 입력 상태 전역 변수
is_pressed_left  = False
is_pressed_right = False
is_pressed_space = False
is_pressed_shift = False
is_pressed_esc   = False
steering_wheel = 0
gas            = 0
break_system   = 0

def key_press(key, mod):
    global is_pressed_left, is_pressed_right, is_pressed_space, is_pressed_shift, is_pressed_esc
    if key == 65361:
        is_pressed_left = True
    if key == 65363:
        is_pressed_right = True
    if key == 32:
        is_pressed_space = True
    if key == 65505:
        is_pressed_shift = True
    if key == 65307:
        is_pressed_esc = True

def key_release(key, mod):
    global is_pressed_left, is_pressed_right, is_pressed_space, is_pressed_shift
    if key == 65361:
        is_pressed_left = False
    if key == 65363:
        is_pressed_right = False
    if key == 32:
        is_pressed_space = False
    if key == 65505:
        is_pressed_shift = False

def update_action():
    global steering_wheel, gas, break_system
    if is_pressed_left ^ is_pressed_right:
        if is_pressed_left:
            steering_wheel = max(steering_wheel - 0.1, -1)
        if is_pressed_right:
            steering_wheel = min(steering_wheel + 0.1, 1)
    else:
        if abs(steering_wheel) < 0.1:
            steering_wheel = 0
        elif steering_wheel > 0:
            steering_wheel -= 0.1
        else:
            steering_wheel += 0.1
    if is_pressed_space:
        gas = min(gas + 0.1, 1)
    else:
        gas = max(gas - 0.1, 0)
    if is_pressed_shift:
        break_system = min(break_system + 0.1, 1)
    else:
        break_system = max(break_system - 0.1, 0)

if __name__ == '__main__':
    env = gym.make('CarRacing-v2', render_mode="human")
    state, _ = env.reset()
    env.unwrapped.viewer.window.on_key_press = key_press
    env.unwrapped.viewer.window.on_key_release = key_release

    counter = 0
    total_reward = 0
    while not is_pressed_esc:
        env.render()
        update_action()
        action = [steering_wheel, gas, break_system]
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        counter += 1
        total_reward += reward
        print('Action:[{:+.1f}, {:+.1f}, {:+.1f}] Reward: {:.3f}'.format(action[0], action[1], action[2], reward))
        if done:
            print("Restart game after {} timesteps. Total Reward: {}".format(counter, total_reward))
            counter = 0
            total_reward = 0
            state, _ = env.reset()
            continue

    env.close()
