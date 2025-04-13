# DQN_PER_HUMAN

PER 사용:

python train_model.py -s 1 -e 1000 -p 1.0 --per 1

Uniform Replay 사용 (PER 미사용):

python train_model.py -s 1 -e 1000 -p 1.0 --per 0

모델 평가/플레이 실행

python play_car_racing_by_the_model.py -m ./save/trial_400.pth -e 5

학습 결과 비교 (그래프)

python plot_training.py
