# train_model.py
import numpy as np
# NumPy 최신 버전에서 np.bool8 없으면 패치
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

import random
import torch

# 랜덤 seed 설정 (재현성을 위해 동일 seed 사용)
seed = 42  # 원하는 seed 값
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

import argparse
import gym
import csv
from collections import deque
from CarRacingDQNAgent import CarRacingDQNAgent
from common_functions import process_state_image, generate_state_frame_stack_from_queue

RENDER = True
STARTING_EPISODE = 1
ENDING_EPISODE = 1000
SKIP_FRAMES = 2
SAVE_TRAINING_FREQUENCY = 25
UPDATE_TARGET_MODEL_FREQUENCY = 5

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a DQN agent to play CarRacing using PyTorch.')
    parser.add_argument('-m', '--model', help='이전에 저장된 모델 파일 경로')
    parser.add_argument('-s', '--start', type=int, default=1, help='시작 에피소드 (기본값 1)')
    parser.add_argument('-e', '--end', type=int, default=1000, help='종료 에피소드 (기본값 1000)')
    parser.add_argument('-p', '--epsilon', type=float, default=1.0, help='시작 epsilon (기본값 1.0)')
    parser.add_argument('--per', type=int, default=1, help='PER 사용 여부: 1 (사용), 0 (미사용)')
    args = parser.parse_args()

    # 환경 생성 시 render_mode 지정4 (랜더 안할려면 "rgb_array"로 변경, 하려면 "human"으로 변경)
    env = gym.make('CarRacing-v2', render_mode="rgb_array")
    use_per = bool(args.per)
    agent = CarRacingDQNAgent(epsilon=args.epsilon, use_per=use_per)
    if args.model:
        agent.load(args.model)
    STARTING_EPISODE = args.start
    ENDING_EPISODE = args.end

    training_rewards = []  # 에피소드별 총 보상 기록

    for e in range(STARTING_EPISODE, ENDING_EPISODE + 1):
        # Gym API: reset()이 (observation, info) 튜플 반환 → 첫 번째 값 사용
        init_state, _ = env.reset()
        init_state = process_state_image(init_state)

        total_reward = 0
        negative_reward_counter = 0
        state_frame_stack_queue = deque([init_state] * agent.frame_stack_num, maxlen=agent.frame_stack_num)
        time_frame_counter = 1
        done = False

        while True:
            if RENDER:
                env.render()

            current_state_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
            action = agent.act(current_state_stack)

            reward = 0
            # Gym v2: step() → (observation, reward, terminated, truncated, info)
            for _ in range(SKIP_FRAMES + 1):
                next_state, r, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                reward += r
                if done:
                    break

            negative_reward_counter = negative_reward_counter + 1 if time_frame_counter > 100 and reward < 0 else 0

            # 보너스: 가속만 사용할 때 보상 증가
            if action[1] == 1 and action[2] == 0:
                reward *= 1.5

            total_reward += reward

            next_state_processed = process_state_image(next_state)
            state_frame_stack_queue.append(next_state_processed)
            next_state_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)

            agent.memorize(current_state_stack, action, reward, next_state_stack, done)

            if done or negative_reward_counter >= 25 or total_reward < 0:
                print('Episode: {}/{}, Time Frames: {}, Total Reward(adjusted): {:.2f}, Epsilon: {:.2f}'
                      .format(e, ENDING_EPISODE, time_frame_counter, total_reward, agent.epsilon))
                break

            agent.replay()
            time_frame_counter += 1

        training_rewards.append([e, total_reward])

        if e % UPDATE_TARGET_MODEL_FREQUENCY == 0:
            agent.update_target_model()

        if e % SAVE_TRAINING_FREQUENCY == 0:
            agent.save('./save/trial_{}.pth'.format(e))

    env.close()

    # 학습 로그를 CSV 파일로 저장 (PER 사용 여부에 따라 파일명 구분)
    log_filename = "training_log_PER.csv" if use_per else "training_log_uniform.csv"
    with open(log_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["episode", "total_reward"])
        writer.writerows(training_rewards)
    print("Training log saved to", log_filename)
