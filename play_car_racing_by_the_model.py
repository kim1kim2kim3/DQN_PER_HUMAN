# play_car_racing_by_the_model.py
import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

import argparse
import gym
from collections import deque
from CarRacingDQNAgent import CarRacingDQNAgent
from common_functions import process_state_image, generate_state_frame_stack_from_queue

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Play CarRacing using the trained PyTorch model.')
    parser.add_argument('-m', '--model', required=True, help='학습된 모델 파일 (예: trial_400.pth)')
    parser.add_argument('-e', '--episodes', type=int, default=1, help='플레이할 에피소드 수')
    args = parser.parse_args()
    
    env = gym.make('CarRacing-v2', render_mode="human")
    agent = CarRacingDQNAgent(epsilon=0)  # epsilon=0: 탐험 없이 모델 예측만 사용
    agent.load(args.model)

    for e in range(args.episodes):
        init_state, _ = env.reset()
        init_state = process_state_image(init_state)

        total_reward = 0
        state_frame_stack_queue = deque([init_state] * agent.frame_stack_num, maxlen=agent.frame_stack_num)
        time_frame_counter = 1
        
        while True:
            env.render()
            current_state_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
            action = agent.act(current_state_stack)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            next_state_processed = process_state_image(next_state)
            state_frame_stack_queue.append(next_state_processed)

            if done:
                print('Episode: {}/{}, Time Frames: {}, Total Reward: {:.2f}'
                      .format(e + 1, args.episodes, time_frame_counter, total_reward))
                break
            time_frame_counter += 1

    env.close()
